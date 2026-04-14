"""
Framework-neutral block-sparsity data structures.

BlockSparseTensors (cute.Tensor based) is defined here for use by root-level
CuTe DSL kernel files (e.g. block_sparse_utils.py).

All torch-specific helpers (BlockSparseTensorsTorch, normalize_block_sparse_config,
to_cute_block_sparse_tensors, etc.) are re-exported from torch.block_sparsity
so that torch/interface.py and torch/compute_block_sparsity.py can import them
from the canonical root path  flash_mask.flash_attn_v4.block_sparsity.
"""

from flash_mask.flash_attn_v4.torch.block_sparsity import (
    BlockSparseTensors,
    BlockSparseTensorsTorch,
    ceildiv,
    get_sparse_q_block_size,
    get_block_sparse_expected_shapes,
    get_block_sparse_expected_shapes_bwd,
    infer_block_sparse_expected_shapes,
    normalize_block_sparse_tensors,
    normalize_block_sparse_config,
    normalize_block_sparse_config_bwd,
    is_block_sparsity_enabled,
    get_block_sparse_broadcast_pattern,
    to_cute_block_sparse_tensors,
    fast_sampling,
    _expand_sparsity_tensor,
    _check_and_expand_block,
)

__all__ = [
    "BlockSparseTensors",
    "BlockSparseTensorsTorch",
    "ceildiv",
    "get_sparse_q_block_size",
    "get_block_sparse_expected_shapes",
    "get_block_sparse_expected_shapes_bwd",
    "infer_block_sparse_expected_shapes",
    "normalize_block_sparse_tensors",
    "normalize_block_sparse_config",
    "normalize_block_sparse_config_bwd",
    "is_block_sparsity_enabled",
    "get_block_sparse_broadcast_pattern",
    "to_cute_block_sparse_tensors",
    "fast_sampling",
]
