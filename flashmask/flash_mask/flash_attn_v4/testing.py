# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

"""
Minimal root-level testing utilities.

Provides is_fake_mode() for torch/interface.py which does:
    from flash_mask.flash_attn_v4.testing import is_fake_mode
"""


def is_fake_mode() -> bool:
    """Return True if currently inside a torch FakeTensorMode context."""
    try:
        from torch._guards import active_fake_mode
        return active_fake_mode() is not None
    except ImportError:
        return False
