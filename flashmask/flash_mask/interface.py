# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flash Attention v4 unified public interface.

Exposes the framework-specific backend through a single stable import path:

    from flash_mask.interface import flash_attn_func, flash_attn_varlen_func

The backend is fixed at **install time** via the ``FLASH_ATTN_BACKEND``
environment variable passed to ``pip install`` / ``setup.py``:

    FLASH_ATTN_BACKEND=paddle pip install -e . --no-build-isolation   # default
    FLASH_ATTN_BACKEND=torch  pip install -e . --no-build-isolation

The chosen value is baked into ``flash_mask/_backend.py`` by ``setup.py`` so
that no environment variable is needed at runtime.
"""

import importlib

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_combine",
]

_BACKENDS = {
    "paddle": "flash_mask.flash_attn_v4.paddle.interface",
    "torch":  "flash_mask.flash_attn_v4.torch.interface",
}

# ---------------------------------------------------------------------------
# Resolve backend name — compile-time first, runtime env-var as fallback
# (the fallback exists only for development checkouts where setup.py has not
# been run yet; production installs always have _backend.py)
# ---------------------------------------------------------------------------
try:
    from flash_mask._backend import BACKEND as _backend_name
except ImportError:
    import os as _os
    _backend_name = _os.environ.get("FLASH_ATTN_BACKEND", "paddle").lower()

if _backend_name not in _BACKENDS:
    raise ValueError(
        f"Unknown backend {_backend_name!r}, "
        f"expected one of {sorted(_BACKENDS)}"
    )

try:
    _mod = importlib.import_module(_BACKENDS[_backend_name])
except ImportError as e:
    raise ImportError(
        f"Backend {_backend_name!r} is not available. "
        f"Re-install with FLASH_ATTN_BACKEND={_backend_name} to confirm, "
        f"or switch backend via FLASH_ATTN_BACKEND at install time."
    ) from e

flash_attn_func        = _mod.flash_attn_func
flash_attn_varlen_func = _mod.flash_attn_varlen_func
flash_attn_combine     = _mod.flash_attn_combine
