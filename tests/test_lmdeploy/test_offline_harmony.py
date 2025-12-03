"""Test GPT-OSS harmony format parsing in offline pipeline.

This test verifies that the offline pipeline correctly parses GPT-OSS harmony
format output into separate content, reasoning_content, and tool_calls fields.
"""

import pytest


def test_gpt_oss_offline_parsing():
    """Test that offline pipeline parses harmony format for GPT-OSS models."""
    pytest.skip("Requires GPT-OSS model to be available")

    import lmdeploy

    # This test would require an actual GPT-OSS model
    pipe = lmdeploy.pipeline("openai/gpt-oss-20b")
    response = pipe(["What is 2+2?"])

    # Should have parsed content (not raw harmony tokens)
    assert response.text
    assert "<|channel|>" not in response.text
    assert "<|message|>" not in response.text

    # Should have reasoning content if model generated it
    # (may be None if model didn't use reasoning for this simple question)
    assert hasattr(response, "reasoning_content")

    # Should have tool_calls attribute (may be None if no tools called)
    assert hasattr(response, "tool_calls")


def test_harmony_parsing_can_be_disabled():
    """Test that harmony parsing can be disabled to get raw tokens."""
    pytest.skip("Requires GPT-OSS model to be available")

    import lmdeploy

    pipe = lmdeploy.pipeline("openai/gpt-oss-20b")

    # Use generate with parse_harmony=False
    # Note: batch_infer doesn't expose parse_harmony yet, need to use generate directly
    import asyncio

    async def test_raw():
        response_gen = pipe.engine.generate(
            ["test"], session_id=1, parse_harmony=False, stream_response=False
        )
        response = None
        async for r in response_gen:
            response = r
        return response

    response = asyncio.run(test_raw())

    # Should have raw harmony tokens
    assert "<|channel|>" in response.response or "<|start|>" in response.response


def test_response_dataclass_has_harmony_fields():
    """Test that Response dataclass has harmony format fields."""
    from lmdeploy.messages import Response

    # Create a Response instance
    response = Response(
        text="test",
        generate_token_len=1,
        input_token_len=1,
        reasoning_content="reasoning",
        tool_calls=[{"name": "test"}],
    )

    assert response.text == "test"
    assert response.reasoning_content == "reasoning"
    assert response.tool_calls == [{"name": "test"}]


def test_genout_dataclass_has_harmony_fields():
    """Test that GenOut dataclass has harmony format fields."""
    from lmdeploy.serve.async_engine import GenOut

    # Create a GenOut instance
    genout = GenOut(
        response="test",
        history_token_len=0,
        input_token_len=1,
        generate_token_len=1,
        reasoning_content="reasoning",
        tool_calls=[{"name": "test"}],
    )

    assert genout.response == "test"
    assert genout.reasoning_content == "reasoning"
    assert genout.tool_calls == [{"name": "test"}]


if __name__ == "__main__":
    # Run basic dataclass tests
    test_response_dataclass_has_harmony_fields()
    test_genout_dataclass_has_harmony_fields()
    print("✓ Dataclass tests passed")
