"""
Tests for per-sequence planned draft lengths semantics.

These are pure-Python unit tests that mirror the high-level logic used in
LlamaBatch for EAGLE per-sequence draft/accepted/rewind lengths, without
requiring a built TurboMind extension. They intentionally focus on the
per-sequence accounting (planned_tokens_per_seq[b], accepted_len[b],
rewind_len[b]) rather than invoking the full decode loop.
"""

import pytest


def compute_kv_lengths(planned_tokens_per_seq, accepted_lens):
    """Mirror the C++ logic for KV draft/accepted/rewind lengths."""
    assert len(planned_tokens_per_seq) == len(accepted_lens)
    kv_draft = []
    kv_accept = []
    kv_rewind = []
    for planned, acc in zip(planned_tokens_per_seq, accepted_lens):
        planned = max(1, planned)
        kv_draft_len = planned
        kv_accepted_len = max(1, min(acc, kv_draft_len))
        rewind_len = max(0, kv_draft_len - kv_accepted_len)
        kv_draft.append(kv_draft_len)
        kv_accept.append(kv_accepted_len)
        kv_rewind.append(rewind_len)
    return kv_draft, kv_accept, kv_rewind


def test_per_sequence_planned_tokens_affect_rewind_lengths():
    """Heterogeneous planned tokens per seq yield per-seq rewind lengths."""
    planned = [4, 2, 3]
    accepted = [3, 1, 0]

    kv_draft, kv_accept, kv_rewind = compute_kv_lengths(planned, accepted)

    # Sequence 0: draft=4, accepted=3 -> rewind 1.
    assert kv_draft[0] == 4
    assert kv_accept[0] == 3
    assert kv_rewind[0] == 1

    # Sequence 1: draft=2, accepted=1 -> rewind 1.
    assert kv_draft[1] == 2
    assert kv_accept[1] == 1
    assert kv_rewind[1] == 1

    # Sequence 2: draft=3, accepted=0 -> we still keep 1 token and rewind 2.
    assert kv_draft[2] == 3
    assert kv_accept[2] == 1
    assert kv_rewind[2] == 2

