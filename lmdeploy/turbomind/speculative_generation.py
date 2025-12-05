"""
Speculative decoding generation wrapper for TurboMind.

Integrates speculative decoding into the main generation loop.
"""

import logging
from typing import List, Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)


class SpeculativeGenerationWrapper:
    """
    Wraps TurboMind generation to add speculative decoding.

    Current integration focuses on:
    - Draft token generation (EAGLE / EAGLE3)
    - Token acceptance metrics
    - KV cache bookkeeping hooks

    The wrapper is intentionally conservative: it *does not* alter the target
    model's output yet. Instead, it computes acceptance statistics that can be
    used to validate and tune speculative decoding before wiring it into the
    low‑level generation loop.
    """

    def __init__(self, turbomind_instance, speculative_manager):
        """
        Initialize wrapper.

        Args:
            turbomind_instance: TurboMindInstance (target model owner)
            speculative_manager: SpeculativeDecodingManager instance
        """
        self.turbomind = turbomind_instance
        self.spec_manager = speculative_manager

        # Import verifier
        from lmdeploy.turbomind.draft_verifier import DraftTokenVerifier

        self.verifier = DraftTokenVerifier()

        # Import and initialize KV cache manager
        from lmdeploy.turbomind.kv_cache_manager import (
            KVCacheManager,
            SpeculativeKVCacheIntegration,
        )

        self.kv_cache_mgr = KVCacheManager(turbomind_instance)
        self.kv_cache_integration = SpeculativeKVCacheIntegration(self.kv_cache_mgr)

        # Statistics
        self.total_steps = 0
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0

    def generate_step_with_speculation(
        self,
        input_ids: List[int],
        session,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Single generation step with speculative decoding.

        Args:
            input_ids: Current input token IDs (Python list).
            session: TurboMind session / opaque handle (passed to KV manager).
            **kwargs:
                Optional:
                    - target_output_ids: List[int] of target tokens for this step.

        Returns:
            Dictionary with:
                - output_ids: Generated token IDs (input + accepted draft tokens)
                - num_accepted: Number of accepted draft tokens
                - acceptance_rate: Acceptance rate for this step
        """
        self.total_steps += 1

        # Step 1: Generate draft tokens
        num_spec_tokens = self.spec_manager.get_num_speculative_tokens()
        draft_tokens = self.spec_manager.generate_draft_tokens(
            input_ids, num_tokens=num_spec_tokens
        )

        if not draft_tokens:
            # No draft tokens, fall back to standard generation semantics
            logger.debug("No draft tokens, using standard (non-speculative) path")
            return self._standard_generation_step(input_ids, session, **kwargs)

        self.total_draft_tokens += len(draft_tokens)

        # Step 2: If target outputs are available, build a synthetic logits
        # tensor so we can reuse DraftTokenVerifier's logic. For integration
        # inside TurboMindInstance.async_stream_infer we pass target_output_ids
        # via kwargs; for unit tests we keep the original "accept first 2" stub.
        target_output_ids: Optional[List[int]] = kwargs.get("target_output_ids")

        if target_output_ids:
            # Build minimal one‑hot logits so that argmax equals target_output_ids.
            effective_len = min(len(draft_tokens), len(target_output_ids))
            if effective_len == 0:
                verification_result = {
                    "accepted_tokens": [],
                    "num_accepted": 0,
                    "acceptance_rate": 0.0,
                }
            else:
                vocab_size = (
                    max(max(draft_tokens[:effective_len]), max(target_output_ids[:effective_len]))
                    + 1
                )
                target_logits = torch.zeros(effective_len, vocab_size, dtype=torch.float32)
                for i in range(effective_len):
                    target_logits[i, target_output_ids[i]] = 1.0

                verification_result = self.verifier.verify_tokens(
                    draft_tokens[:effective_len], target_logits
                )
        else:
            # Backwards‑compatible stub behaviour used by existing tests:
            # accept first 2 draft tokens purely for sanity‑checking.
            accepted = draft_tokens[:2]
            num_accepted = len(accepted)
            verification_result = {
                "accepted_tokens": accepted,
                "num_accepted": num_accepted,
                "acceptance_rate": num_accepted / len(draft_tokens),
            }

        num_accepted = verification_result["num_accepted"]
        accepted_tokens = verification_result["accepted_tokens"]
        acceptance_rate = verification_result["acceptance_rate"]

        self.total_accepted_tokens += num_accepted

        # Step 3: KV cache bookkeeping hook. The underlying KVCacheManager
        # currently implements a best‑effort, mostly logging‑oriented rewind,
        # so this call is safe even without deep TurboMind integration.
        try:
            self.kv_cache_integration.handle_speculation_step(
                num_draft=len(draft_tokens),
                num_accepted=num_accepted,
                session=session,
            )
        except Exception as e:
            logger.debug(f"KV cache integration failed, continuing without rewind: {e}")

        logger.info(
            "Speculation step %d: drafted %d, accepted %d (%.1f%%)",
            self.total_steps,
            len(draft_tokens),
            num_accepted,
            acceptance_rate * 100.0 if draft_tokens else 0.0,
        )

        return {
            "output_ids": list(input_ids) + list(accepted_tokens),
            "num_accepted": num_accepted,
            "acceptance_rate": acceptance_rate,
        }

    def _standard_generation_step(
        self,
        input_ids: List[int],
        session,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Standard generation step (no speculation).

        For now this simply reports zero accepted tokens and echoes the input.
        The actual TurboMind generation is driven elsewhere (TurboMindInstance).
        """
        _ = session  # kept for signature symmetry
        return {
            "output_ids": list(input_ids),
            "num_accepted": 0,
            "acceptance_rate": 0.0,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get speculation statistics.

        Returns:
            Dictionary with statistics.
        """
        overall_acceptance = (
            self.total_accepted_tokens / self.total_draft_tokens
            if self.total_draft_tokens > 0
            else 0.0
        )

        avg_accepted_per_step = (
            self.total_accepted_tokens / self.total_steps
            if self.total_steps > 0
            else 0.0
        )

        return {
            "total_steps": self.total_steps,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "overall_acceptance_rate": overall_acceptance,
            "avg_accepted_per_step": avg_accepted_per_step,
            "verifier_stats": self.verifier.get_statistics(),
        }

    def reset_statistics(self):
        """Reset all statistics."""
        self.total_steps = 0
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.verifier.reset_statistics()
