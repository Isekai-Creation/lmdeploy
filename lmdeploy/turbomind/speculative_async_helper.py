"""
Speculative decoding helper for TurboMind async integration.

Provides synchronous wrapper for async inference to enable
speculative decoding in the generation loop.
"""

import asyncio
import logging
from typing import List, Optional
import torch

logger = logging.getLogger(__name__)


def run_async_inference_sync(async_func, *args, **kwargs):
    """
    Run an async function synchronously.
    
    This is needed because TurboMindInstance.async_stream_infer is async
    but we need to call it from synchronous code.
    
    Args:
        async_func: Async function to run
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Result of async function
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context
            # Create a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, async_func(*args, **kwargs)
                )
                return future.result()
        else:
            # No running loop, we can use asyncio.run
            return asyncio.run(async_func(*args, **kwargs))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(async_func(*args, **kwargs))


async def generate_draft_tokens_async(
    draft_model_instance,
    input_ids: List[int],
    num_tokens: int,
    session_id: int = 0
) -> List[int]:
    """
    Generate draft tokens asynchronously using TurboMind.
    
    Args:
        draft_model_instance: TurboMindInstance for draft model
        input_ids: Input token IDs
        num_tokens: Number of tokens to generate
        session_id: Session ID for inference
        
    Returns:
        List of draft token IDs
    """
    from lmdeploy.messages import GenerationConfig
    
    # Use greedy decoding for draft
    gen_config = GenerationConfig(
        max_new_tokens=num_tokens,
        temperature=0.0,  # Greedy
        top_k=1,
        top_p=1.0,
    )
    
    draft_tokens = []
    
    try:
        # Run async inference
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
            
            # Stop if we have enough tokens
            if len(draft_tokens) >= num_tokens:
                draft_tokens = draft_tokens[:num_tokens]
                break
        
        logger.debug(f\"Generated {len(draft_tokens)} draft tokens: {draft_tokens}\")
        return draft_tokens
        
    except Exception as e:
        logger.error(f\"Async draft generation failed: {e}\")\
        import traceback
        logger.error(traceback.format_exc())
        return []


def generate_draft_tokens_sync(
    draft_model_instance,
    input_ids: List[int],
    num_tokens: int,
    session_id: int = 0
) -> List[int]:
    """
    Generate draft tokens synchronously (wrapper for async version).
    
    Args:
        draft_model_instance: TurboMindInstance for draft model
        input_ids: Input token IDs
        num_tokens: Number of tokens to generate
        session_id: Session ID for inference
        
    Returns:
        List of draft token IDs
    """
    return run_async_inference_sync(
        generate_draft_tokens_async,
        draft_model_instance,
        input_ids,
        num_tokens,
        session_id
    )
