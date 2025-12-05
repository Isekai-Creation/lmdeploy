"""
Python helpers for speculative decoding kernels.

This module provides a thin, torch-based wrapper around the semantics of
the KV cache rewind kernel implemented in
``lmdeploy/lmdeploy/turbomind/kernels/speculative_decoding/common.{h,cu}``.

It is primarily intended for tests such as
``lmdeploy/tests/turbomind/test_speculative_kernels.py`` and mirrors the
behaviour of ``KVCacheRewindParams`` + ``invokeKVCacheRewind`` using
torch tensor operations on CPU or CUDA devices.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class KVCacheRewindParams:
    """Host-side description of KV cache rewind parameters.

    This mirrors the C++ ``KVCacheRewindParams`` struct enough for test
    purposes. Tensors should be 1D/2D as documented in the C++ header:

    - ``rewind_lengths``: [max_batch_size]
    - ``batch_slots``: [batch_size] (optional)
    - ``block_tables``: [max_batch_size, max_blocks_per_seq]
    - ``kv_cache_blocks``: optional; when provided, a tensor of shape
      [num_layers, max_blocks_per_seq] whose rows will have their tail
      entries cleared in sync with ``block_tables``.
    """

    kv_cache_blocks: Optional[torch.Tensor]
    rewind_lengths: torch.Tensor
    batch_slots: Optional[torch.Tensor]
    block_tables: torch.Tensor
    batch_size: int
    max_batch_size: int
    num_layers: int
    block_size: int
    max_blocks_per_seq: int


def invoke_kv_cache_rewind(params: KVCacheRewindParams) -> None:
    """Apply KV cache rewind semantics to the provided block tables.

    Behaviour mirrors the CUDA kernel in ``common.cu``:

        - For each ``batch_idx`` in ``[0, batch_size)``:
            - Map to a global ``slot`` via ``batch_slots`` when present,
              otherwise use ``slot = batch_idx`` directly.
            - Look up ``rewind_len = rewind_lengths[slot]``.
            - Compute ``blocks_to_free = ceil(rewind_len / block_size)``.
            - For that ``slot``, set the last ``blocks_to_free`` entries
              in ``block_tables[slot, :]`` to ``-1``.
            - If ``kv_cache_blocks`` is provided, clear the corresponding
              entries as well.

    All tensor operations are performed in-place on the input tensors.
    """

    block_tables = params.block_tables
    rewind_lengths = params.rewind_lengths
    batch_slots = params.batch_slots
    kv_cache_blocks = params.kv_cache_blocks

    batch_size = int(params.batch_size)
    max_blocks_per_seq = int(params.max_blocks_per_seq)
    block_size = int(params.block_size)
    num_layers = int(params.num_layers)

    if batch_size <= 0:
        return

    device = block_tables.device

    if batch_slots is None:
        slots = torch.arange(batch_size, device=device, dtype=torch.int32)
    else:
        slots = batch_slots.to(device=device, dtype=torch.int32)[:batch_size]

    rewind_lengths = rewind_lengths.to(device=device)

    for batch_idx in range(batch_size):
        slot = int(slots[batch_idx].item())
        if slot < 0 or slot >= params.max_batch_size:
            continue

        rewind_len = int(rewind_lengths[slot].item())
        if rewind_len <= 0:
            continue

        blocks_to_free = (rewind_len + block_size - 1) // block_size
        if blocks_to_free <= 0:
            continue

        # Clear tail entries in the block table row for this slot.
        start = max(0, max_blocks_per_seq - blocks_to_free)
        block_tables[slot, start:max_blocks_per_seq] = -1

        # Optionally clear the corresponding kv_cache_blocks entries.
        if kv_cache_blocks is not None:
            for layer in range(num_layers):
                kv_cache_blocks[layer, start:max_blocks_per_seq] = 0


def benchmark_kv_cache_rewind(
    batch_size: int = 4,
    max_batch_size: int = 4,
    max_blocks_per_seq: int = 8,
    num_layers: int = 2,
    block_size: int = 16,
    iters: int = 50,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Microbenchmark for the KV rewind helper.

    This runs ``invoke_kv_cache_rewind`` repeatedly on synthetic inputs and
    reports simple timing statistics. It uses torch timing (CPU or CUDA
    events) but does not depend on the full TurboMind engine.
    """
    import time

    if iters <= 0:
        raise ValueError("iters must be positive")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    max_batch_size_t = max_batch_size

    block_tables = (
        torch.arange(max_batch_size_t * max_blocks_per_seq, dtype=torch.int32, device=device)
        .view(max_batch_size_t, max_blocks_per_seq)
        .clone()
    )
    rewind_lengths = torch.zeros(max_batch_size_t, dtype=torch.int32, device=device)
    batch_slots = torch.arange(batch_size, dtype=torch.int32, device=device)
    kv_cache_blocks = torch.ones(num_layers, max_blocks_per_seq, dtype=torch.int32, device=device)

    # Synthetic pattern: first half of slots rewind one block, second half two blocks.
    for i in range(batch_size):
        rewind_lengths[i] = block_size * (1 + (i % 2))

    params = KVCacheRewindParams(
        kv_cache_blocks=kv_cache_blocks,
        rewind_lengths=rewind_lengths,
        batch_slots=batch_slots,
        block_tables=block_tables,
        batch_size=batch_size,
        max_batch_size=max_batch_size_t,
        num_layers=num_layers,
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
    )

    # Warmup
    invoke_kv_cache_rewind(params)
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(iters):
        # Reset block tables between iterations to avoid early exits.
        block_tables.copy_(
            torch.arange(max_batch_size_t * max_blocks_per_seq, dtype=torch.int32, device=device).view(
                max_batch_size_t, max_blocks_per_seq
            )
        )
        invoke_kv_cache_rewind(params)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    avg_s_per_call = elapsed / float(iters)
    total_rewinds = int(batch_size * iters)
    rewinds_per_second = total_rewinds / elapsed if elapsed > 0 else 0.0

    return {
        "batch_size": float(batch_size),
        "max_blocks_per_seq": float(max_blocks_per_seq),
        "num_layers": float(num_layers),
        "block_size": float(block_size),
        "iters": float(iters),
        "avg_ms_per_call": avg_s_per_call * 1000.0,
        "rewinds_per_second": rewinds_per_second,
    }

