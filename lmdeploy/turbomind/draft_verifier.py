"""
Draft token verification for speculative decoding.

Implements token-by-token verification of draft tokens against target model outputs.
"""

import torch
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DraftTokenVerifier:
    """
    Verifies draft tokens against target model outputs.

    Implements the core acceptance logic for speculative decoding:
    - Token-by-token comparison
    - Probability-based acceptance (optional)
    - Acceptance statistics tracking
    """

    def __init__(self, use_probabilistic=False, temperature=1.0):
        """
        Initialize verifier.

        Args:
            use_probabilistic: Use probability-based acceptance
            temperature: Temperature for probabilistic acceptance
        """
        self.use_probabilistic = use_probabilistic
        self.temperature = temperature

        # Statistics
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0

    def verify_tokens(
        self,
        draft_tokens: List[int],
        target_logits: torch.Tensor,
        draft_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Verify draft tokens against target model logits.

        Args:
            draft_tokens: List of draft token IDs
            target_logits: Target model logits [num_tokens, vocab_size]
            draft_logits: Draft model logits (optional, for probabilistic)

        Returns:
            Dictionary with:
                - accepted_tokens: List of accepted token IDs
                - num_accepted: Number of accepted tokens
                - acceptance_rate: Acceptance rate for this batch
        """
        if len(draft_tokens) == 0:
            return {"accepted_tokens": [], "num_accepted": 0, "acceptance_rate": 0.0}

        # Greedy verification (default)
        if not self.use_probabilistic or draft_logits is None:
            return self._verify_greedy(draft_tokens, target_logits)
        else:
            return self._verify_probabilistic(draft_tokens, target_logits, draft_logits)

    def _verify_greedy(
        self, draft_tokens: List[int], target_logits: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Greedy verification: accept if draft token matches target's argmax.

        Args:
            draft_tokens: List of draft token IDs
            target_logits: Target model logits [num_tokens, vocab_size]

        Returns:
            Verification results
        """
        accepted_tokens = []

        # Get target model's greedy predictions
        target_predictions = target_logits.argmax(dim=-1).tolist()

        # Accept tokens until first mismatch
        for i, draft_token in enumerate(draft_tokens):
            if i >= len(target_predictions):
                break

            if draft_token == target_predictions[i]:
                accepted_tokens.append(draft_token)
            else:
                # First mismatch - stop accepting
                break

        num_accepted = len(accepted_tokens)
        acceptance_rate = num_accepted / len(draft_tokens) if draft_tokens else 0.0

        # Update statistics
        self.total_draft_tokens += len(draft_tokens)
        self.total_accepted_tokens += num_accepted

        logger.debug(
            f"Verified {len(draft_tokens)} draft tokens, "
            f"accepted {num_accepted} ({acceptance_rate:.1%})"
        )

        return {
            "accepted_tokens": accepted_tokens,
            "num_accepted": num_accepted,
            "acceptance_rate": acceptance_rate,
        }

    def _verify_probabilistic(
        self,
        draft_tokens: List[int],
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Probabilistic verification using rejection sampling.

        Based on: https://arxiv.org/abs/2211.17192 (Speculative Sampling)

        Args:
            draft_tokens: List of draft token IDs
            target_logits: Target model logits [num_tokens, vocab_size]
            draft_logits: Draft model logits [num_tokens, vocab_size]

        Returns:
            Verification results
        """
        accepted_tokens = []

        # Convert logits to probabilities
        target_probs = torch.softmax(target_logits / self.temperature, dim=-1)
        draft_probs = torch.softmax(draft_logits / self.temperature, dim=-1)

        # Verify each token
        for i, draft_token in enumerate(draft_tokens):
            if i >= len(target_probs):
                break

            # Get probabilities for draft token
            p_target = target_probs[i, draft_token].item()
            p_draft = draft_probs[i, draft_token].item()

            # Acceptance probability: min(1, p_target / p_draft)
            accept_prob = min(1.0, p_target / (p_draft + 1e-10))

            # Sample acceptance
            if torch.rand(1).item() < accept_prob:
                accepted_tokens.append(draft_token)
            else:
                # Rejection - stop here
                break

        num_accepted = len(accepted_tokens)
        acceptance_rate = num_accepted / len(draft_tokens) if draft_tokens else 0.0

        # Update statistics
        self.total_draft_tokens += len(draft_tokens)
        self.total_accepted_tokens += num_accepted

        logger.debug(
            f"Probabilistic verification: {len(draft_tokens)} draft tokens, "
            f"accepted {num_accepted} ({acceptance_rate:.1%})"
        )

        return {
            "accepted_tokens": accepted_tokens,
            "num_accepted": num_accepted,
            "acceptance_rate": acceptance_rate,
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        Get overall acceptance statistics.

        Returns:
            Dictionary with statistics
        """
        overall_rate = (
            self.total_accepted_tokens / self.total_draft_tokens
            if self.total_draft_tokens > 0
            else 0.0
        )

        return {
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "overall_acceptance_rate": overall_rate,
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
