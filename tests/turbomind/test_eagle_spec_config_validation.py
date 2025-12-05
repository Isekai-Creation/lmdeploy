import pytest

from lmdeploy.speculative_config import SpeculativeConfig


def test_eagle_spec_config_rejects_zero_path_len():
    """
    EAGLE should reject a configuration where max_path_len <= 0.
    """
    with pytest.raises(ValueError, match="max_path_len"):
        SpeculativeConfig(
            method="eagle",
            model="dummy",
            num_speculative_tokens=3,
            max_path_len=0,
        )


def test_eagle_spec_config_rejects_negative_path_len():
    """
    EAGLE should reject a configuration where max_path_len < 0.
    """
    with pytest.raises(ValueError, match="max_path_len"):
        SpeculativeConfig(
            method="eagle",
            model="dummy",
            num_speculative_tokens=3,
            max_path_len=-1,
        )


def test_eagle_spec_config_rejects_zero_decoding_tokens():
    """
    EAGLE should reject max_decoding_tokens <= 0.
    """
    with pytest.raises(ValueError, match="max_decoding_tokens"):
        SpeculativeConfig(
            method="eagle",
            model="dummy",
            num_speculative_tokens=3,
            max_path_len=4,
            max_decoding_tokens=0,
        )


def test_eagle_spec_config_rejects_path_len_exceeding_decoding_tokens():
    """
    max_path_len must be <= max_decoding_tokens.
    """
    with pytest.raises(ValueError, match="max_path_len"):
        SpeculativeConfig(
            method="eagle",
            model="dummy",
            num_speculative_tokens=3,
            max_path_len=16,
            max_decoding_tokens=8,
        )


def test_eagle_spec_config_rejects_path_len_smaller_than_num_spec_tokens():
    """
    max_path_len must be >= num_speculative_tokens so the tree can
    represent the speculative step.
    """
    with pytest.raises(ValueError, match="max_path_len"):
        SpeculativeConfig(
            method="eagle3",
            model="dummy",
            num_speculative_tokens=8,
            max_path_len=4,
            max_decoding_tokens=16,
        )


def test_eagle_spec_config_rejects_non_positive_max_non_leaves():
    """
    max_non_leaves_per_layer must be positive.
    """
    with pytest.raises(ValueError, match="max_non_leaves_per_layer"):
        SpeculativeConfig(
            method="eagle3",
            model="dummy",
            num_speculative_tokens=4,
            max_path_len=4,
            max_decoding_tokens=8,
            max_non_leaves_per_layer=0,
        )


def test_eagle_spec_config_accepts_minimal_valid_config_and_sets_defaults():
    """
    Positive sanity check: a minimal valid config should not raise and
    should populate EAGLE-specific defaults.
    """
    cfg = SpeculativeConfig(
        method="eagle3",
        model="dummy",
        num_speculative_tokens=3,
    )

    # Basic structural invariants
    assert cfg.method == "eagle3"
    assert cfg.num_speculative_tokens == 3
    assert cfg.max_path_len > 0
    assert cfg.max_decoding_tokens > 0
    assert cfg.max_non_leaves_per_layer > 0
    assert cfg.max_path_len >= cfg.num_speculative_tokens
    assert cfg.max_path_len <= cfg.max_decoding_tokens
