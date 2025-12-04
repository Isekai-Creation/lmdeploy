#!/usr/bin/env python3
"""
Regression tests for the verification layer (Phase 2).
Focus on execute_verification pass/fail parsing.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inline_tir_engine import InlineTIREngine


class _DummyEngine:
    """Placeholder async engine; unused in these tests."""

    pass


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.engine = InlineTIREngine(
            async_engine=_DummyEngine(), code_timeout=2.0, max_code_output_len=512
        )
        os.makedirs("logs", exist_ok=True)

    def test_range_constraint_failure(self):
        """Out-of-range candidate should fail verification and report reason."""
        verification_code = """
value = 150
if 0 <= value <= 100:
    print("VERIFIED")
else:
    print("FAILED: out of range")
"""
        result = self.engine.execute_verification(
            verification_code=verification_code,
            answer="150",
            question_id="range_constraint",
        )

        self.assertFalse(result["verified"])
        self.assertIn("out of range", result["reason"])

    def test_prime_verification_success(self):
        """Correct prime candidate should be verified."""
        verification_code = """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

value = 29
if is_prime(value) and sum(1 for i in range(2, value+1) if is_prime(i)) == 10:
    print("VERIFIED")
else:
    print("FAILED: not 10th prime")
"""
        result = self.engine.execute_verification(
            verification_code=verification_code,
            answer="29",
            question_id="prime_success",
        )

        self.assertTrue(result["verified"])
        self.assertEqual(result["reason"], "")

    def test_verdict_parser_variants(self):
        """Test the internal verdict parser for robustness."""
        parse = self.engine._parse_verification_output

        # Single VERIFIED
        v = parse("some log\nVERIFIED\nmore log\n")
        self.assertEqual(v["status"], "ok")
        self.assertTrue(v["verified"])

        # Single FAILED
        v = parse("FAILED: out of range\n")
        self.assertEqual(v["status"], "ok")
        self.assertFalse(v["verified"])
        self.assertIn("out of range", v["reason"])

        # Duplicate VERIFIED lines
        v = parse("VERIFIED\nVERIFIED\n")
        self.assertTrue(v["verified"])

        # Duplicate FAILED with same reason
        v = parse("FAILED: reason\nFAILED: reason\n")
        self.assertFalse(v["verified"])
        self.assertIn("reason", v["reason"])

        # Conflicting FAILED reasons
        v = parse("FAILED: r1\nFAILED: r2\n")
        self.assertFalse(v["verified"])
        self.assertEqual(v["status"], "conflicting")

        # No verdict
        v = parse("some output\nno verdict here\n")
        self.assertFalse(v["verified"])
        self.assertEqual(v["status"], "malformed")

    def test_classify_tool_flavor_verification(self):
        code = '''
candidate = 42
if True:
    print("VERIFIED")
else:
    print("FAILED: out of range")
'''
        flavor = self.engine._classify_tool_flavor(code)
        self.assertEqual(flavor, "verification")

    def test_classify_tool_flavor_normal(self):
        code = "x = 1 + 2\nprint(x)"
        flavor = self.engine._classify_tool_flavor(code)
        self.assertEqual(flavor, "normal")

    def test_summarize_inline_verification_picks_last_verified(self):
        history = [
            {
                "code": "print('FAILED: wrong')",
                "result": "FAILED: wrong\n",
                "flavor": "verification",
            },
            {
                "code": "print('VERIFIED')",
                "result": "some log\nVERIFIED\nmore\n",
                "flavor": "verification",
            },
        ]
        summary = self.engine._summarize_inline_verification(history)
        self.assertTrue(summary["inline_verified"])
        meta = summary["inline_verification_meta"]
        self.assertIsNotNone(meta)
        self.assertEqual(meta["status"], "ok")
        self.assertTrue(meta["verified"])

    def test_summarize_inline_verification_no_verification_calls(self):
        history = [
            {"code": "x = 1+1", "result": "2\n", "flavor": "normal"},
        ]
        summary = self.engine._summarize_inline_verification(history)
        self.assertFalse(summary["inline_verified"])
        self.assertIsNone(summary["inline_verification_meta"])


if __name__ == "__main__":
    unittest.main()
