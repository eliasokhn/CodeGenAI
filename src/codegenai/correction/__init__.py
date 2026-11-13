"""Feedback-based self-correction loop."""
from codegenai.correction.feedback import assert_no_hidden_leak, format_feedback
from codegenai.correction.loop import Attempt, CorrectionTrace, SelfCorrectionLoop

__all__ = ["Attempt", "CorrectionTrace", "SelfCorrectionLoop", "assert_no_hidden_leak", "format_feedback"]
