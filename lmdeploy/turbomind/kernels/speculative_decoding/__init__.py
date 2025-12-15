"""
Python wrappers for speculative decoding kernels.

This package currently exposes a small set of helper APIs used in tests:

- ``common``: host-side KV rewind wrapper that mirrors the behaviour of
  the C++/CUDA ``KVCacheRewindParams`` + ``invokeKVCacheRewind`` kernel.

The goal is to keep these wrappers lightweight and aligned with the C++
implementations, while still being usable from Python-only test code.
"""

from . import common

__all__ = ["common"]

