"""
Cross-backend EAGLE3 SpeculativeConfig alignment tests.

These tests exercise the Python-facing parts of both backends to ensure
that a given SpeculativeConfig:

  - can be passed to both PytorchEngineConfig and TurbomindEngineConfig
    without violating their validation logic,
  - yields consistent core semantics (method and num_speculative_tokens),
  - and maps cleanly onto the engine-side speculative_config dict used by
    TurboMind via SpeculativeConfig.to_turbomind_spec_dict().
"""

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.speculative_config import SpeculativeConfig


def _make_eagle3_cfg() -> SpeculativeConfig:
    return SpeculativeConfig(
        method="eagle3",
        model="draft-model",
        num_speculative_tokens=4,
        max_path_len=7,
        max_decoding_tokens=16,
        max_non_leaves_per_layer=12,
    )


def test_spec_config_works_for_both_backends_minimal():
    """A single EAGLE3 SpeculativeConfig should be valid for both backends."""
    spec_cfg = _make_eagle3_cfg()

    # Pytorch backend: config should be accepted as-is.
    pt_cfg = PytorchEngineConfig(
        tp=1,
        session_len=1024,
        max_batch_size=1,
        cache_max_entry_count=0.8,
        speculative_config=spec_cfg,
    )
    assert pt_cfg.speculative_config is spec_cfg

    # TurboMind backend: config should also be accepted.
    tm_cfg = TurbomindEngineConfig(
        tp=1,
        session_len=1024,
        max_batch_size=1,
        cache_max_entry_count=0.8,
        cache_block_seq_len=64,
        speculative_config=spec_cfg,
    )
    assert tm_cfg.speculative_config is spec_cfg

    # Core semantics must be identical.
    assert spec_cfg.method == "eagle3"
    assert spec_cfg.num_speculative_tokens == 4


def test_spec_config_to_turbomind_dict_matches_engine_fields():
    """SpeculativeConfig.to_turbomind_spec_dict should match TurboMind keys."""
    spec_cfg = _make_eagle3_cfg()

    spec_dict = spec_cfg.to_turbomind_spec_dict()

    # This dict is what the C++ Triton backend reads as speculative_config.
    assert spec_dict["method"] == "eagle3"
    assert spec_dict["model"] == "draft-model"
    assert spec_dict["num_speculative_tokens"] == 4
    assert spec_dict["max_path_len"] == 7
    assert spec_dict["max_decoding_tokens"] == 16
    assert spec_dict["max_non_leaves_per_layer"] == 12

