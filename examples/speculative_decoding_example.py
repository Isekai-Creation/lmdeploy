"""
Example usage of speculative decoding with LMDeploy TurboMind.

Demonstrates how to use Draft/Target and EAGLE3 speculative decoding.
"""

from lmdeploy import TurboMind
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.speculative_config import SpeculativeConfig


def example_draft_target():
    """Example: Draft/Target speculative decoding."""
    print("=" * 80)
    print("Example 1: Draft/Target Speculative Decoding")
    print("=" * 80)

    # Configure speculative decoding
    spec_config = SpeculativeConfig(
        method="draft_target",
        model="/path/to/draft/model",  # Smaller, faster model
        num_speculative_tokens=3,
    )

    # Configure engine with speculation
    engine_config = TurbomindEngineConfig(tp=1, speculative_config=spec_config)

    # Create TurboMind instance
    tm = TurboMind(model_path="/path/to/target/model", engine_config=engine_config)

    print(f"Speculative decoding enabled: {tm.speculative_manager is not None}")
    if tm.speculative_manager:
        print(f"Method: {tm.speculative_manager.get_method()}")
        print(
            f"Num speculative tokens: {tm.speculative_manager.get_num_speculative_tokens()}"
        )

    # Generate (speculation happens automatically)
    # outputs = tm.generate(prompts=["What is AI?"], max_new_tokens=50)
    print("\nNote: Actual generation requires draft model implementation")


def example_eagle3():
    """Example: EAGLE3 speculative decoding."""
    print("\n" + "=" * 80)
    print("Example 2: EAGLE3 Speculative Decoding")
    print("=" * 80)

    # Configure EAGLE3
    spec_config = SpeculativeConfig(
        method="eagle",
        model="/path/to/eagle/draft/model",
        num_speculative_tokens=5,
        max_path_len=5,
        max_decoding_tokens=10,
        max_non_leaves_per_layer=10,
        capture_layers=[-1],  # Capture last layer
    )

    # Configure engine
    engine_config = TurbomindEngineConfig(tp=1, speculative_config=spec_config)

    # Create TurboMind instance
    tm = TurboMind(model_path="/path/to/target/model", engine_config=engine_config)

    print(f"Speculative decoding enabled: {tm.speculative_manager is not None}")
    if tm.speculative_manager:
        print(f"Method: {tm.speculative_manager.get_method()}")
        print(
            f"Num speculative tokens: {tm.speculative_manager.get_num_speculative_tokens()}"
        )
        print(f"EAGLE max_path_len: {spec_config.max_path_len}")
        print(f"EAGLE max_decoding_tokens: {spec_config.max_decoding_tokens}")

    print("\nNote: Actual generation requires EAGLE3 implementation")


def example_ngram():
    """Example: NGram speculative decoding."""
    print("\n" + "=" * 80)
    print("Example 3: NGram Speculative Decoding")
    print("=" * 80)

    # Configure NGram (no draft model needed!)
    spec_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=3,
        max_matching_ngram_size=4,
        is_public_pool=True,
    )

    # Configure engine
    engine_config = TurbomindEngineConfig(tp=1, speculative_config=spec_config)

    # Create TurboMind instance
    tm = TurboMind(model_path="/path/to/target/model", engine_config=engine_config)

    print(f"Speculative decoding enabled: {tm.speculative_manager is not None}")
    if tm.speculative_manager:
        print(f"Method: {tm.speculative_manager.get_method()}")
        print(
            f"Num speculative tokens: {tm.speculative_manager.get_num_speculative_tokens()}"
        )
        print(f"NGram max_matching_ngram_size: {spec_config.max_matching_ngram_size}")

    print("\nNote: Actual generation requires NGram implementation")


def example_disabled():
    """Example: TurboMind without speculative decoding."""
    print("\n" + "=" * 80)
    print("Example 4: No Speculative Decoding (Default)")
    print("=" * 80)

    # No speculative_config specified
    engine_config = TurbomindEngineConfig(tp=1)

    tm = TurboMind(model_path="/path/to/model", engine_config=engine_config)

    print(f"Speculative decoding enabled: {tm.speculative_manager is not None}")
    print("Running in standard (non-speculative) mode")


if __name__ == "__main__":
    print("LMDeploy Speculative Decoding Examples\n")

    # Note: These examples show the API usage
    # Actual model paths need to be provided

    try:
        example_draft_target()
    except Exception as e:
        print(f"Draft/Target example error: {e}")

    try:
        example_eagle3()
    except Exception as e:
        print(f"EAGLE3 example error: {e}")

    try:
        example_ngram()
    except Exception as e:
        print(f"NGram example error: {e}")

    try:
        example_disabled()
    except Exception as e:
        print(f"Disabled example error: {e}")

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
