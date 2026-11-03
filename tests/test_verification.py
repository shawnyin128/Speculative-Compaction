"""Tests for the DREAM-R style line verifier (spec-compact online gate)."""

import math

import pytest

from compaction.verification import (
    LineVerdict,
    build_judge_messages,
    hard_gate,
    positive_score,
    verdict_from_top_logprobs,
)

# --- positive_score: the four boundary classes from the design doc -----------


def test_positive_score_normal_case():
    top = [("positive", -0.1), ("negative", -2.5)]
    score = positive_score(top)
    expected = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.5))
    assert score == pytest.approx(expected)


def test_positive_score_leading_space_and_case_variants():
    top = [(" Positive", -0.2), ("NEGATIVE", -1.0)]
    score = positive_score(top)
    expected = math.exp(-0.2) / (math.exp(-0.2) + math.exp(-1.0))
    assert score == pytest.approx(expected)


def test_positive_score_missing_both_returns_none():
    assert positive_score([("yes", -0.1), ("no", -1.0)]) is None


def test_positive_score_one_sided_is_bounded_not_forced():
    """One label present scores against an upper bound on the absent one.

    The absent label can hold at most the probability mass the top-k list
    leaves unlisted, so the score reflects how much the present label held
    instead of collapsing to exactly 1.0 / 0.0.
    """
    assert positive_score([(" positive", -0.3)]) == pytest.approx(math.exp(-0.3))
    assert positive_score([("negative", -0.3)]) == pytest.approx(1 - math.exp(-0.3))


def test_positive_score_one_sided_bound_uses_the_smallest_listed_logprob():
    # 0.49 of the mass is unlisted, but an absent token cannot outrank a listed
    # one, so "negative" is bounded by the 0.01 floor instead.
    top = [("positive", math.log(0.5)), ("other", math.log(0.01))]
    assert positive_score(top) == pytest.approx(0.5 / (0.5 + 0.01))


def test_confident_judge_with_negative_off_the_list_accepts():
    """A peaked judge (Gemma4-31B) answers positive at logprob ~0 and never
    lists "negative" in its top-20; that is its most confident accept."""
    top = [("positive", -1e-6)] + [(f"tok{i}", -16.0 - 0.2 * i) for i in range(19)]
    verdict = verdict_from_top_logprobs(top, eta=0.7)
    assert verdict.accepted and verdict.reason == "judge_accept"
    assert verdict.score > 0.999


def test_confident_negative_with_positive_off_the_list_rejects_as_judge_reject():
    top = [("negative", -1e-6)] + [(f"tok{i}", -16.0 - 0.2 * i) for i in range(19)]
    verdict = verdict_from_top_logprobs(top, eta=0.7)
    assert not verdict.accepted and verdict.reason == "judge_reject"
    assert verdict.score < 1e-3


def test_positive_score_rejects_when_judge_did_not_answer():
    """Both labels present but the argmax is neither -- the judge answered
    something else, so its preference between two tokens it barely considered
    is not evidence."""
    top = [("The", -0.2), ("positive", -12.0), ("negative", -13.0)]
    assert positive_score(top) is None


def test_positive_score_unanswered_judge_would_have_forced_accept():
    """Regression pin for the bug's actual shape: the pathological list scores
    1.0 under the old one-sided arithmetic, i.e. accept at any eta <= 1."""
    pathological = [("The", -0.2), ("I", -1.5), ("positive", -12.0)]
    old_pos = math.exp(-12.0)
    assert old_pos / (old_pos + 0.0) == pytest.approx(1.0)
    verdict = verdict_from_top_logprobs(pathological, 0.7)
    assert verdict.accepted is False
    assert verdict.reason == "judge_tokens_missing"
    assert verdict.score is None


def test_positive_score_tie_is_half():
    assert positive_score([("positive", -1.0), ("negative", -1.0)]) == pytest.approx(0.5)


def test_positive_score_keeps_best_logprob_per_token():
    # Duplicate surface forms (e.g. " positive" and "positive") keep the max.
    top = [("positive", -3.0), (" positive", -0.5), ("negative", -1.0)]
    expected = math.exp(-0.5) / (math.exp(-0.5) + math.exp(-1.0))
    assert positive_score(top) == pytest.approx(expected)


# --- verdict_from_top_logprobs: threshold + missing-token policy -------------


def test_verdict_accepts_at_threshold():
    v = verdict_from_top_logprobs([("positive", -0.1), ("negative", -5.0)], eta=0.7)
    assert isinstance(v, LineVerdict)
    assert v.accepted and v.reason == "judge_accept"
    assert v.score is not None and v.score > 0.9


def test_verdict_rejects_below_threshold():
    v = verdict_from_top_logprobs([("positive", -3.0), ("negative", -0.1)], eta=0.7)
    assert not v.accepted and v.reason == "judge_reject"


def test_verdict_missing_tokens_rejects_without_crash():
    v = verdict_from_top_logprobs([("yes", -0.1)], eta=0.7)
    assert not v.accepted
    assert v.reason == "judge_tokens_missing"
    assert v.score is None


def test_verdict_boundary_exactly_eta_accepts():
    # score == eta must accept (>= semantics; boundary-value lesson).
    v = verdict_from_top_logprobs([("positive", -1.0), ("negative", -1.0)], eta=0.5)
    assert v.accepted


# --- hard_gate ----------------------------------------------------------------


def test_hard_gate_accepts_normal_line():
    ok, reason = hard_gate("- Keep the total under $650.", ["- Book the flight."], 400)
    assert ok and reason == "ok"


def test_hard_gate_rejects_empty_and_whitespace():
    assert hard_gate("", [], 400) == (False, "empty_line")
    assert hard_gate("   \t", [], 400) == (False, "empty_line")


def test_hard_gate_rejects_duplicate_normalized():
    accepted = ["- Keep the total under $650."]
    ok, reason = hard_gate("-   keep the total under $650.", accepted, 400)
    assert not ok and reason == "duplicate_line"


def test_hard_gate_rejects_overlong():
    ok, reason = hard_gate("x" * 401, [], 400)
    assert not ok and reason == "line_too_long"


# --- build_judge_messages ------------------------------------------------------


def test_judge_messages_contain_all_parts_and_single_word_instruction():
    messages = build_judge_messages(
        context_text="[user] book a flight under $650",
        slot="constraints_preferences",
        accepted_lines=["- Budget cap $650."],
        proposed_line="- Prefers aisle seats.",
    )
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]
    for fragment in [
        "[user] book a flight under $650",
        "constraints_preferences",
        "- Budget cap $650.",
        "- Prefers aisle seats.",
        "positive",
        "negative",
    ]:
        assert fragment in content
    # No accepted lines yet must also render without crashing.
    empty = build_judge_messages("ctx", "goal", [], "- line")
    assert "goal" in empty[0]["content"]


# --- sr judge: the same accept-direction failure, on the 0-9 scorer ----------


def test_digit_score_normal_case():
    from compaction.verification import digit_expectation_score

    top = [("8", -0.1), ("7", -2.0)]
    w8, w7 = math.exp(-0.1), math.exp(-2.0)
    assert digit_expectation_score(top) == pytest.approx((8 * w8 + 7 * w7) / (w8 + w7))


def test_digit_score_rejects_when_judge_did_not_answer():
    """A single stray digit collapses the expectation onto that digit exactly,
    regardless of its probability -- a '9' at rank 18 scored a perfect 9.0 and
    cleared the sr threshold of 7.0."""
    from compaction.verification import digit_expectation_score

    pathological = [("The", -0.2), ("Sure", -1.1), ("9", -13.8)]
    assert digit_expectation_score(pathological) is None
    verdict = verdict_from_top_logprobs(pathological, 7.0, verify_method="sr")
    assert verdict.accepted is False
    assert verdict.reason == "judge_tokens_missing"


def test_digit_score_answered_low_still_scores():
    """The guard must not swallow a genuine low score: a judge that answered
    '2' still returns ~2.0 and is rejected on the threshold, not as missing."""
    from compaction.verification import digit_expectation_score

    top = [("2", -0.1), ("3", -2.0)]
    assert digit_expectation_score(top) is not None
    verdict = verdict_from_top_logprobs(top, 7.0, verify_method="sr")
    assert verdict.accepted is False
    assert verdict.reason == "judge_reject"
