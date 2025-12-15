"""
Speculative decoding helper for TurboMind async integration.

Provides a synchronous wrapper around the async inference API so that
draft generation can be triggered from synchronous code paths.
"""

import asyncio
import concurrent.futures
import logging
from typing import List

from lmdeploy.messages import GenerationConfig

logger = logging.getLogger(__name__)


def run_async_inference_sync(async_func, *args, **kwargs):
    """Run an async function synchronously.

    This is useful when TurboMindInstance.async_stream_infer (async) needs to
    be called from a synchronous context.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop – offload to a worker thread.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, async_func(*args, **kwargs))
            return future.result()

    # No running loop, we can use asyncio.run directly.
    return asyncio.run(async_func(*args, **kwargs))


async def generate_draft_tokens_async(
    draft_model_instance,
    input_ids: List[int],
    num_tokens: int,
    session_id: int = 0,
) -> List[int]:
    """Generate draft tokens asynchronously using a TurboMindInstance.

    Args:
        draft_model_instance: TurboMindInstance for the draft model.
        input_ids: Input token IDs.
        num_tokens: Number of tokens to generate.
        session_id: Session ID for inference.

    Returns:
        List of draft token IDs.
    """
    # Greedy decoding for the draft model.
    gen_config = GenerationConfig(
        max_new_tokens=num_tokens,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
    )

    draft_tokens: List[int] = []

    try:
        async for output in draft_model_instance.async_stream_infer(
            session_id=session_id,
            input_ids=input_ids,
            sequence_start=True,
            sequence_end=True,
            gen_config=gen_config,
            stream_output=False,
        ):
            if output.token_ids:
                draft_tokens.extend(output.token_ids)

            if len(draft_tokens) >= num_tokens:
                draft_tokens = draft_tokens[:num_tokens]
                break

        logger.debug(
            "Generated %d draft tokens from async draft model: %s",
            len(draft_tokens),
            draft_tokens,
        )
        return draft_tokens
    except Exception as e:
        import traceback

        logger.error("Async draft generation failed: %s", e)
        logger.debug(traceback.format_exc())
        return []


def generate_draft_tokens_sync(
    draft_model_instance,
    input_ids: List[int],
    num_tokens: int,
    session_id: int = 0,
) -> List[int]:
    """Synchronous wrapper for :func:`generate_draft_tokens_async`."""
    return run_async_inference_sync(
        generate_draft_tokens_async,
        draft_model_instance,
        input_ids,
        num_tokens,
        session_id,
    )

