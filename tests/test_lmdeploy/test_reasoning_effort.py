"""Test reasoning_effort support in GenerationConfig."""


def test_reasoning_effort_in_generation_config():
    """Test that GenerationConfig accepts reasoning_effort parameter."""
    from lmdeploy.messages import GenerationConfig

    # Test with high reasoning
    config = GenerationConfig(reasoning_effort='high')
    assert config.reasoning_effort == 'high'

    # Test with low reasoning
    config = GenerationConfig(reasoning_effort='low')
    assert config.reasoning_effort == 'low'

    # Test with medium reasoning
    config = GenerationConfig(reasoning_effort='medium')
    assert config.reasoning_effort == 'medium'

    # Test default (None)
    config = GenerationConfig()
    assert config.reasoning_effort is None

    print('✓ All reasoning_effort tests passed')


def test_reasoning_effort_with_other_params():
    """Test reasoning_effort works with other GenerationConfig parameters."""
    from lmdeploy.messages import GenerationConfig

    config = GenerationConfig(
        temperature=1.0,
        top_p=0.9,
        max_new_tokens=1000,
        reasoning_effort='high',
        skip_special_tokens=True,
    )

    assert config.reasoning_effort == 'high'
    assert config.temperature == 1.0
    assert config.top_p == 0.9
    assert config.max_new_tokens == 1000
    assert config.skip_special_tokens is True

    print('✓ Combined parameters test passed')


if __name__ == '__main__':
    test_reasoning_effort_in_generation_config()
    test_reasoning_effort_with_other_params()
    print('\n✅ All tests passed!')
