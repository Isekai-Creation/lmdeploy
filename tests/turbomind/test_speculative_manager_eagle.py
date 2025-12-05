"""
Unit tests for SpeculativeDecodingManager EAGLE configuration.

These tests ensure that when method is 'eagle' or 'eagle3', the Python
speculative manager defers entirely to native TurboMind EAGLE and does
not attempt to run a separate Python draft model.
"""

from lmdeploy.speculative_config import SpeculativeConfig
from lmdeploy.turbomind.speculative_manager import SpeculativeDecodingManager


def test_eagle_uses_native_turbomind_and_no_python_draft():
    """SpeculativeDecodingManager should not create a Python draft model for EAGLE."""
    cfg = SpeculativeConfig(
        method="eagle3",
        model="dummy-draft-model",
        num_speculative_tokens=3,
    )

    mgr = SpeculativeDecodingManager(cfg, target_model_comm=None)

    assert mgr.get_method() == "eagle3"
    # EAGLE/EAGLE3: native TurboMind owns the draft model; Python draft_model stays None.
    assert getattr(mgr, "draft_model", None) is None

    # generate_draft_tokens should be a no-op and return an empty list.
    draft_tokens = mgr.generate_draft_tokens([1, 2, 3])
    assert draft_tokens == []

