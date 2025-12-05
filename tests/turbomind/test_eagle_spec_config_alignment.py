"""
Alignment tests for SpeculativeConfig vs TurboMind engine speculative_config.

These tests ensure that:
  - SpeculativeConfig.to_turbomind_spec_dict() produces the expected
    engine-side speculative_config mapping.
  - check_turbomind_spec_alignment emits a warning when an engine-side
    config drifts from the Python SpeculativeConfig.
"""

import warnings

from lmdeploy.speculative_config import (
    SpeculativeConfig,
    check_turbomind_spec_alignment,
)


def test_to_turbomind_spec_dict_matches_expected_keys():
    """to_turbomind_spec_dict should expose the fields EngineParam uses."""
    cfg = SpeculativeConfig(
        method="eagle3",
        model="draft-model",
        num_speculative_tokens=4,
        max_path_len=7,
        max_decoding_tokens=16,
        max_non_leaves_per_layer=12,
    )

    spec_dict = cfg.to_turbomind_spec_dict()

    assert spec_dict["method"] == "eagle3"
    assert spec_dict["model"] == "draft-model"
    assert spec_dict["num_speculative_tokens"] == 4
    assert spec_dict["max_path_len"] == 7
    assert spec_dict["max_decoding_tokens"] == 16
    assert spec_dict["max_non_leaves_per_layer"] == 12


def test_check_turbomind_spec_alignment_no_warning_when_matching():
    """check_turbomind_spec_alignment should be silent on matching configs."""
    cfg = SpeculativeConfig(
        method="eagle",
        model="draft",
        num_speculative_tokens=3,
        max_path_len=5,
        max_decoding_tokens=10,
        max_non_leaves_per_layer=8,
    )
    engine_spec = cfg.to_turbomind_spec_dict().copy()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_turbomind_spec_alignment(cfg, engine_spec)
        assert not w, "No warnings expected when specs match"


def test_check_turbomind_spec_alignment_warns_on_mismatch():
    """A mismatch between Python and engine speculative_config should warn."""
    cfg = SpeculativeConfig(
        method="eagle3",
        model="draft",
        num_speculative_tokens=3,
        max_path_len=6,
        max_decoding_tokens=12,
        max_non_leaves_per_layer=10,
    )
    engine_spec = cfg.to_turbomind_spec_dict().copy()
    # Introduce a deliberate mismatch in max_path_len.
    engine_spec["max_path_len"] = cfg.max_path_len + 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_turbomind_spec_alignment(cfg, engine_spec)
        assert w, "Expected a warning for mismatched speculative_config"
        msg = str(w[0].message)
        assert "max_path_len" in msg

