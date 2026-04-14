"""
Framework-neutral CuTe DSL utilities.

Re-exports everything from torch.cute_dsl_utils so that torch/block_sparsity.py,
torch/interface.py, and other modules that do
  from flash_mask.flash_attn_v4.cute_dsl_utils import ...
continue to work without modification.
"""

from flash_mask.flash_attn_v4.torch.cute_dsl_utils import (
    StaticTypes,
    cute_compile_patched,
    load_cubin_module_data_patched,
    get_max_active_clusters,
    get_device_capacity,
    assume_strides_aligned,
    assume_tensor_aligned,
    to_cute_tensor,
    to_cute_aux_tensor,
    get_aux_tensor_metadata,
    get_broadcast_dims,
    ParamsBase,
    ArgumentsBase,
    make_fake_tensor,
    torch2cute_dtype_map,
)

__all__ = [
    "StaticTypes",
    "cute_compile_patched",
    "load_cubin_module_data_patched",
    "get_max_active_clusters",
    "get_device_capacity",
    "assume_strides_aligned",
    "assume_tensor_aligned",
    "to_cute_tensor",
    "to_cute_aux_tensor",
    "get_aux_tensor_metadata",
    "get_broadcast_dims",
    "ParamsBase",
    "ArgumentsBase",
    "make_fake_tensor",
    "torch2cute_dtype_map",
]
