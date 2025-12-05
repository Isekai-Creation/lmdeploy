"""
Additional tests for EAGLE behaviour in the Python speculative managers.

These tests ensure that when method is 'eagle' or 'eagle3', both the
single-request SpeculativeDecodingManager and the batch-aware managers
do not attempt to run a Python draft model and instead defer entirely
to native TurboMind EAGLE.
"""

from lmdeploy.speculative_config import SpeculativeConfig
from lmdeploy.turbomind.batch_speculative_manager import (
    BatchSpeculativeManager,
    OptimizedBatchSpeculativeManager,
)


def _make_eagle_cfg() -> SpeculativeConfig:
    return SpeculativeConfig(
        method="eagle3",
        model="dummy-draft-model",
        num_speculative_tokens=3,
    )


def test_batch_manager_eagle_uses_native_turbomind_and_no_python_draft():
    """BatchSpeculativeManager should return empty drafts for EAGLE."""
    cfg = _make_eagle_cfg()
    mgr = BatchSpeculativeManager(cfg, target_model_comm=None)

    batch_input_ids = [[1, 2, 3], [4, 5]]
    drafts = mgr.generate_batch_draft_tokens_sync(batch_input_ids)

    assert isinstance(drafts, list)
    assert len(drafts) == len(batch_input_ids)
    assert all(d == [] for d in drafts)


def test_optimized_batch_manager_eagle_uses_native_turbomind_and_no_python_draft():
    """OptimizedBatchSpeculativeManager should also return empty drafts for EAGLE."""
    cfg = _make_eagle_cfg()
    mgr = OptimizedBatchSpeculativeManager(cfg, target_model_comm=None, batch_size=4)

    batch_input_ids = [[1, 2, 3], [4, 5, 6]]
    drafts = mgr.generate_batch_draft_tokens_sync(batch_input_ids)

    assert isinstance(drafts, list)
    assert len(drafts) == len(batch_input_ids)
    assert all(d == [] for d in drafts)

