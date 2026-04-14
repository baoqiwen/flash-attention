"""
Block-sparsity utilities for FlexAttention — Paddle backend.

Mirrors torch/block_sparsity.py but uses paddle.Tensor instead of torch.Tensor.
"""

from typing import Callable, NamedTuple, Tuple

import cutlass.cute as cute
import paddle

from flash_mask.flash_attn_v4.paddle.cute_dsl_utils import get_broadcast_dims, to_cute_tensor

# Re-use the CuTe-tensor based BlockSparseTensors from the root module
from flash_mask.flash_attn_v4.block_sparsity import BlockSparseTensors


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class BlockSparseTensorsPaddle(NamedTuple):
    """Block-sparse tensor container for the Paddle backend."""
    mask_block_cnt: paddle.Tensor
    mask_block_idx: paddle.Tensor
    full_block_cnt: paddle.Tensor | None = None
    full_block_idx: paddle.Tensor | None = None
    block_size: tuple[int, int] | None = None


def get_sparse_q_block_size(
    tensors: BlockSparseTensorsPaddle | None,
    seqlen_q: int,
) -> int | None:
    """Return the Q sparse block size, or None when sparsity is unset or ambiguous."""
    if tensors is None:
        return None
    if tensors.block_size is not None:
        return tensors.block_size[0]
    num_m_blocks = tensors.mask_block_idx.shape[2]
    min_block_size = ceildiv(seqlen_q, num_m_blocks)
    max_block_size = seqlen_q if num_m_blocks == 1 else (seqlen_q - 1) // (num_m_blocks - 1)
    if min_block_size != max_block_size:
        return None
    return min_block_size


def _expand_sparsity_tensor(
    tensor: paddle.Tensor,
    expected_shape: Tuple[int, ...],
    tensor_name: str,
    context: str | None,
    hint: str | Callable[[], str] | None,
) -> paddle.Tensor:
    """Check if we need to expand the tensor to expected shape, and do so if possible."""
    needs_expand = list(tensor.shape) != list(expected_shape)
    if not needs_expand:
        return tensor
    can_expand = all(map(lambda cur, tgt: cur == tgt or cur == 1, tensor.shape, expected_shape))
    if not can_expand:
        context_clause = f" ({context})" if context else ""
        resolved_hint = hint() if callable(hint) else hint
        hint_clause = f" Hint: {resolved_hint}" if resolved_hint else ""
        raise ValueError(
            f"{tensor_name}{context_clause} with shape {tensor.shape} cannot be expanded to expected shape {expected_shape}."
            f"{hint_clause}"
        )
    return tensor.expand(list(expected_shape))


def _check_and_expand_block(
    name: str,
    cnt: paddle.Tensor | None,
    idx: paddle.Tensor | None,
    expected_count_shape: Tuple[int, int, int],
    expected_index_shape: Tuple[int, int, int, int],
    context: str | None,
    hint: str | Callable[[], str] | None,
) -> Tuple[paddle.Tensor | None, paddle.Tensor | None]:
    if (cnt is None) != (idx is None):
        raise ValueError(
            f"{name}_block_cnt and {name}_block_idx must both be provided or both be None"
        )
    if cnt is None or idx is None:
        return None, None
    if cnt.dtype != paddle.int32 or idx.dtype != paddle.int32:
        raise ValueError(f"{name}_block tensors must have dtype paddle.int32")
    if str(cnt.place) != str(idx.place):
        raise ValueError(f"{name}_block_cnt and {name}_block_idx must be on the same device")
    if not cnt.place.is_gpu_place() or not idx.place.is_gpu_place():
        raise ValueError(f"{name}_block tensors must live on CUDA (GPU)")
    expanded_cnt = _expand_sparsity_tensor(
        cnt, expected_count_shape, f"{name}_block_cnt", context, hint
    )
    # [Note] Allow Compact block sparse indices
    if idx.ndim == 4 and idx.shape[3] <= expected_index_shape[3]:
        expected_index_shape = (*expected_index_shape[:3], idx.shape[3])
    expanded_idx = _expand_sparsity_tensor(
        idx, expected_index_shape, f"{name}_block_idx", context, hint
    )
    return expanded_cnt, expanded_idx


def get_block_sparse_expected_shapes(
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """Return (expected_count_shape, expected_index_shape) for block sparse normalization."""
    m_block_size_effective = q_stage * m_block_size
    expected_m_blocks = ceildiv(seqlen_q, m_block_size_effective)
    expected_n_blocks = ceildiv(seqlen_k, n_block_size)
    expected_count_shape = (batch_size, num_head, expected_m_blocks)
    expected_index_shape = (batch_size, num_head, expected_m_blocks, expected_n_blocks)
    return expected_count_shape, expected_index_shape


def infer_block_sparse_expected_shapes(
    tensors: BlockSparseTensorsPaddle,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
    context: str,
    sparse_block_size_q: int | None = None,
    sparse_block_size_kv: int | None = None,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], int]:
    base_m_block = q_stage * m_block_size
    base_n_block = n_block_size
    if sparse_block_size_kv is None:
        sparse_block_size_kv = base_n_block
    if sparse_block_size_kv != base_n_block:
        raise ValueError(f"Block sparse tensors{context} require BLOCK_SIZE_KV={base_n_block}.")
    if tensors.mask_block_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")
    num_m_blocks = tensors.mask_block_idx.shape[2]

    if sparse_block_size_q is None:
        sparse_block_size_q = get_sparse_q_block_size(tensors, seqlen_q)
        if sparse_block_size_q is None and base_m_block != 1:
            raise ValueError(
                f"Block sparse tensors{context} require explicit sparse_block_size[0] "
                f"to disambiguate block size for seqlen_q={seqlen_q} and num_m_blocks={num_m_blocks}."
            )
        if sparse_block_size_q is None:
            sparse_block_size_q = ceildiv(seqlen_q, num_m_blocks)

    if sparse_block_size_q % base_m_block != 0:
        raise ValueError(
            f"Block sparse tensors{context} have block size {sparse_block_size_q}, "
            f"which must be a multiple of {base_m_block}."
        )

    expected_m_blocks = ceildiv(seqlen_q, sparse_block_size_q)
    expected_n_blocks = ceildiv(seqlen_k, sparse_block_size_kv)
    q_subtile_factor = sparse_block_size_q // base_m_block
    expected_count_shape = (batch_size, num_head, expected_m_blocks)
    expected_index_shape = (batch_size, num_head, expected_m_blocks, expected_n_blocks)

    mask_block_cnt = tensors.mask_block_cnt
    mask_block_idx = tensors.mask_block_idx
    if mask_block_cnt is None or mask_block_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")
    if mask_block_cnt.ndim != 3 or mask_block_idx.ndim != 4:
        raise ValueError(
            f"Block sparse tensors{context} must have shapes (B, H, M) and (B, H, M, N)."
        )
    for dim_name, cur, tgt in (
        ("batch", mask_block_cnt.shape[0], expected_count_shape[0]),
        ("head", mask_block_cnt.shape[1], expected_count_shape[1]),
    ):
        if cur != tgt and cur != 1:
            raise ValueError(f"Block sparse tensors{context} {dim_name} dim must be {tgt} or 1.")
    for dim_name, cur, tgt in (
        ("batch", mask_block_idx.shape[0], expected_index_shape[0]),
        ("head", mask_block_idx.shape[1], expected_index_shape[1]),
    ):
        if cur != tgt and cur != 1:
            raise ValueError(f"Block sparse tensors{context} {dim_name} dim must be {tgt} or 1.")
    if mask_block_cnt.shape[2] != mask_block_idx.shape[2]:
        raise ValueError(f"Block sparse tensors{context} must share the same m-block dimension.")
    if mask_block_idx.shape[3] > expected_n_blocks:
        raise ValueError(
            f"Block sparse tensors{context} n-block dimension must be <= {expected_n_blocks}."
        )
    if expected_m_blocks != num_m_blocks:
        raise ValueError(
            f"Block sparse tensors{context} m-block dimension {num_m_blocks} does not match "
            f"sparse_block_size_q={sparse_block_size_q}. "
            f"Set BlockSparseTensorsPaddle.block_size to match the BlockMask BLOCK_SIZE."
        )
    return expected_count_shape, expected_index_shape, q_subtile_factor


def get_block_sparse_expected_shapes_bwd(
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    subtile_factor: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    sparse_block_size_q = subtile_factor * m_block_size
    expected_m_blocks = ceildiv(seqlen_q, sparse_block_size_q)
    expected_n_blocks = ceildiv(seqlen_k, n_block_size)
    expected_count_shape = (batch_size, num_head, expected_n_blocks)
    expected_index_shape = (batch_size, num_head, expected_n_blocks, expected_m_blocks)
    return expected_count_shape, expected_index_shape


def normalize_block_sparse_tensors(
    tensors: BlockSparseTensorsPaddle,
    *,
    expected_count_shape: Tuple[int, int, int],
    expected_index_shape: Tuple[int, int, int, int],
    context: str | None = None,
    hint: str | Callable[[], str] | None = None,
) -> BlockSparseTensorsPaddle:
    if tensors.mask_block_cnt is None or tensors.mask_block_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")

    mask_cnt, mask_idx = _check_and_expand_block(
        "mask",
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        expected_count_shape,
        expected_index_shape,
        context,
        hint,
    )
    if mask_cnt is None or mask_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")

    full_cnt, full_idx = _check_and_expand_block(
        "full",
        tensors.full_block_cnt,
        tensors.full_block_idx,
        expected_count_shape,
        expected_index_shape,
        context,
        hint,
    )
    if full_cnt is not None and str(mask_cnt.place) != str(full_cnt.place):
        raise ValueError("All block sparse tensors must be on the same device")

    return BlockSparseTensorsPaddle(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=tensors.block_size,
    )


def is_block_sparsity_enabled(tensors: BlockSparseTensorsPaddle) -> bool:
    return any(t is not None for t in (tensors.full_block_cnt, tensors.mask_block_cnt))


def get_block_sparse_broadcast_pattern(
    tensors: BlockSparseTensorsPaddle,
) -> Tuple[Tuple[bool, ...], ...] | None:
    if not is_block_sparsity_enabled(tensors):
        return None

    patterns = []
    for tensor in (
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        tensors.full_block_cnt,
        tensors.full_block_idx,
    ):
        if tensor is not None:
            patterns.append(get_broadcast_dims(tensor))
        else:
            patterns.append(None)
    return tuple(patterns)


def normalize_block_sparse_config(
    tensors: BlockSparseTensorsPaddle,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    q_stage: int,
) -> tuple[BlockSparseTensorsPaddle, Tuple[Tuple[bool, ...], ...] | None, int]:
    m_block_size, n_block_size = block_size
    if tensors.block_size is None:
        sparse_block_size_q, sparse_block_size_kv = None, n_block_size
    else:
        sparse_block_size_q, sparse_block_size_kv = tensors.block_size
    if sparse_block_size_kv != n_block_size:
        raise ValueError(
            f"Block sparsity requires sparse_block_size[1]={n_block_size} to match tile_n."
        )
    expected_count_shape, expected_index_shape, q_subtile_factor = (
        infer_block_sparse_expected_shapes(
            tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            q_stage=q_stage,
            context="forward",
            sparse_block_size_q=sparse_block_size_q,
            sparse_block_size_kv=sparse_block_size_kv,
        )
    )
    normalized_tensors = normalize_block_sparse_tensors(
        tensors,
        expected_count_shape=expected_count_shape,
        expected_index_shape=expected_index_shape,
    )
    return (
        normalized_tensors,
        get_block_sparse_broadcast_pattern(normalized_tensors),
        q_subtile_factor,
    )


def normalize_block_sparse_config_bwd(
    tensors: BlockSparseTensorsPaddle,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    subtile_factor: int,
) -> tuple[BlockSparseTensorsPaddle, Tuple[Tuple[bool, ...], ...] | None]:
    m_block_size, n_block_size = block_size
    if tensors.block_size is None:
        sparse_block_size_q, sparse_block_size_kv = subtile_factor * m_block_size, n_block_size
    else:
        sparse_block_size_q, sparse_block_size_kv = tensors.block_size
    if sparse_block_size_q != subtile_factor * m_block_size:
        raise ValueError(
            f"Block sparsity expects sparse_block_size_q={subtile_factor * m_block_size} "
            f"for subtile_factor={subtile_factor}."
        )
    if sparse_block_size_kv != n_block_size:
        raise ValueError(
            f"Block sparsity expects sparse_block_size[1]={n_block_size} to match tile_n."
        )
    expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes_bwd(
        batch_size,
        num_head,
        seqlen_q,
        seqlen_k,
        m_block_size,
        n_block_size,
        subtile_factor,
    )
    normalized_tensors = normalize_block_sparse_tensors(
        tensors,
        expected_count_shape=expected_count_shape,
        expected_index_shape=expected_index_shape,
        context="_flash_attn_bwd",
        hint=lambda: (
            f"Backward expects Q-direction block-sparse tensors (q_mask_cnt/q_mask_idx, "
            f"and optionally full_q_cnt/full_q_idx). Regenerate the backward BlockMask with "
            f"BLOCK_SIZE=({subtile_factor * m_block_size}, {n_block_size})."
        ),
    )
    return normalized_tensors, get_block_sparse_broadcast_pattern(normalized_tensors)


def to_cute_block_sparse_tensors(
    tensors: BlockSparseTensorsPaddle, enable_tvm_ffi: bool = True
) -> BlockSparseTensors | None:
    """Convert paddle block sparsity tensors to CuTe tensors, optionally for tvm ffi."""
    if not is_block_sparsity_enabled(tensors):
        return None

    (
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        *_,
    ) = tensors

    (
        mask_block_cnt_tensor,
        mask_block_idx_tensor,
    ) = [
        to_cute_tensor(t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi)
        for t in (mask_block_cnt, mask_block_idx)
    ]
    (
        full_block_cnt_tensor,
        full_block_idx_tensor,
    ) = [
        to_cute_tensor(t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi)
        if t is not None
        else None
        for t in (full_block_cnt, full_block_idx)
    ]

    return BlockSparseTensors(
        mask_block_cnt_tensor,
        mask_block_idx_tensor,
        full_block_cnt_tensor,
        full_block_idx_tensor,
    )


def fast_sampling(mask_mod):
    """Convenience decorator to mark mask_mod as safe for 5-point fast sampling."""
    mask_mod.use_fast_sampling = True
    return mask_mod
