"""
Unit tests for speculative generation wrapper.
"""

import pytest
from unittest.mock import Mock, MagicMock
import torch


class TestSpeculativeGenerationWrapper:
    """Tests for SpeculativeGenerationWrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock TurboMind instance
        self.mock_turbomind = Mock()

        # Mock SpeculativeDecodingManager
        self.mock_spec_manager = Mock()
        self.mock_spec_manager.get_num_speculative_tokens.return_value = 3
        self.mock_spec_manager.generate_draft_tokens.return_value = [10, 20, 30]

        # Create wrapper
        from lmdeploy.turbomind.speculative_generation import (
            SpeculativeGenerationWrapper,
        )

        self.wrapper = SpeculativeGenerationWrapper(
            self.mock_turbomind, self.mock_spec_manager
        )

    def test_initialization(self):
        """Test wrapper initialization."""
        assert self.wrapper.turbomind == self.mock_turbomind
        assert self.wrapper.spec_manager == self.mock_spec_manager
        assert self.wrapper.total_steps == 0
        assert self.wrapper.total_draft_tokens == 0

    def test_generate_step_with_draft_tokens(self):
        """Test generation step with draft tokens."""
        input_ids = [1, 2, 3]
        session = Mock()

        result = self.wrapper.generate_step_with_speculation(input_ids, session)

        # Should have called generate_draft_tokens
        self.mock_spec_manager.generate_draft_tokens.assert_called_once()

        # Should return results
        assert "output_ids" in result
        assert "num_accepted" in result
        assert "acceptance_rate" in result

    def test_generate_step_no_draft_tokens(self):
        """Test generation step when no draft tokens generated."""
        input_ids = [1, 2, 3]
        session = Mock()

        # Mock no draft tokens
        self.mock_spec_manager.generate_draft_tokens.return_value = []

        result = self.wrapper.generate_step_with_speculation(input_ids, session)

        # Should fall back to standard generation
        assert result["num_accepted"] == 0

    def test_statistics_tracking(self):
        """Test statistics tracking."""
        input_ids = [1, 2, 3]
        session = Mock()

        # Run a few steps
        for _ in range(3):
            self.wrapper.generate_step_with_speculation(input_ids, session)

        stats = self.wrapper.get_statistics()

        assert stats["total_steps"] == 3
        assert stats["total_draft_tokens"] > 0
        assert "overall_acceptance_rate" in stats

    def test_statistics_reset(self):
        """Test statistics reset."""
        input_ids = [1, 2, 3]
        session = Mock()

        # Run a step
        self.wrapper.generate_step_with_speculation(input_ids, session)

        # Reset
        self.wrapper.reset_statistics()

        assert self.wrapper.total_steps == 0
        assert self.wrapper.total_draft_tokens == 0
        assert self.wrapper.total_accepted_tokens == 0


class TestDraftTokenVerifier:
    """Tests for DraftTokenVerifier."""

    def setup_method(self):
        """Set up test fixtures."""
        from lmdeploy.turbomind.draft_verifier import DraftTokenVerifier

        self.verifier = DraftTokenVerifier()

    def test_greedy_verification_all_match(self):
        """Test greedy verification when all tokens match."""
        draft_tokens = [10, 20, 30]
        target_logits = torch.zeros(3, 100)
        target_logits[0, 10] = 10.0  # Token 10 has highest logit
        target_logits[1, 20] = 10.0  # Token 20 has highest logit
        target_logits[2, 30] = 10.0  # Token 30 has highest logit

        result = self.verifier.verify_tokens(draft_tokens, target_logits)

        assert result["num_accepted"] == 3
        assert result["acceptance_rate"] == 1.0
        assert result["accepted_tokens"] == [10, 20, 30]

    def test_greedy_verification_first_mismatch(self):
        """Test greedy verification with first token mismatch."""
        draft_tokens = [10, 20, 30]
        target_logits = torch.zeros(3, 100)
        target_logits[0, 99] = 10.0  # Different token has highest logit
        target_logits[1, 20] = 10.0
        target_logits[2, 30] = 10.0

        result = self.verifier.verify_tokens(draft_tokens, target_logits)

        assert result["num_accepted"] == 0
        assert result["acceptance_rate"] == 0.0
        assert result["accepted_tokens"] == []

    def test_greedy_verification_partial_match(self):
        """Test greedy verification with partial match."""
        draft_tokens = [10, 20, 30]
        target_logits = torch.zeros(3, 100)
        target_logits[0, 10] = 10.0  # Match
        target_logits[1, 20] = 10.0  # Match
        target_logits[2, 99] = 10.0  # Mismatch

        result = self.verifier.verify_tokens(draft_tokens, target_logits)

        assert result["num_accepted"] == 2
        assert result["acceptance_rate"] == pytest.approx(0.667, rel=0.01)
        assert result["accepted_tokens"] == [10, 20]

    def test_statistics_accumulation(self):
        """Test statistics accumulation across multiple verifications."""
        draft_tokens = [10, 20]
        target_logits = torch.zeros(2, 100)
        target_logits[0, 10] = 10.0
        target_logits[1, 20] = 10.0

        # Run multiple verifications
        for _ in range(5):
            self.verifier.verify_tokens(draft_tokens, target_logits)

        stats = self.verifier.get_statistics()

        assert stats["total_draft_tokens"] == 10  # 2 * 5
        assert stats["total_accepted_tokens"] == 10  # All accepted
        assert stats["overall_acceptance_rate"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
