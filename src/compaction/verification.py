"""DREAM-R style line verifier: the spec-compact ONLINE gate.

The target model judges one proposed summary line at a time with a single
generated token; acceptance is the softmax probability of ``positive`` versus
``negative`` over the returned top-logprobs (mechanism identical to
dream-r/vlm_specreason/spec_reason.py). Deterministic hard gates run before
the judge and cost nothing.

Design contract (docs/review/spec-compact-runtime-design-20260720.md):
- gate reads GENERATED-token top_logprobs only (APC-safe by construction);
- an unanswered judge (no positive/negative in the top-k, or an argmax that is
  neither) is a REJECT plus a counter, never a crash;
- score >= eta accepts (boundary inclusive).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from compaction.prompts import SLOT_CONTRACTS

JUDGE_TOKENS = ("positive", "negative")

JUDGE_INSTRUCTION = (
    "You are verifying one line of a context-compaction summary.\n"
    "Reply 'positive' ONLY if the proposed line is factually supported by the "
    "trajectory AND coherent with the accepted summary lines so far.\n"
    "Reply 'negative' if it contains any factual error, unsupported claim, "
    "contradiction, or content promoted beyond what the trajectory confirms.\n"
    "Reply with exactly one word: positive or negative."
)

# verify_method="sr" (SpecReason, github.com/ruipeterpan/specreason): the same
# gate, scored on a 0-9 scale read as an expected value over digit tokens
# rather than as a binary. Only the instruction and the scorer differ -- the
# message layout below is shared, so both methods keep the trajectory ahead of
# every slot-specific byte and stay cacheable across slots.
SR_JUDGE_INSTRUCTION = (
    "You are verifying one line of a context-compaction summary.\n"
    "Rate how well the proposed line is supported by the trajectory and "
    "coheres with the accepted summary lines so far.\n"
    "Assign a single-digit score from 0 (contradicted or unsupported) to 9 "
    "(fully supported and coherent).\n"
    "Reply with exactly one digit."
)


@dataclass(frozen=True)
class LineVerdict:
    accepted: bool
    score: float | None
    reason: str  # ok-family: judge_accept; reject-family: judge_reject / judge_tokens_missing


def _answered(top_logprobs: list[tuple[str, float]], is_answer) -> bool:
    """Did the judge actually answer the question it was asked?

    Both scorers below renormalize over a SUBSET of the returned distribution,
    which throws away how much mass that subset holds. A judge that replied
    with something else entirely still yields a well-formed score off whatever
    answer token happens to appear far down the list -- and because the scorers
    are scale-free, that score is indistinguishable from a confident one.
    Requiring the argmax to be an answer token restores the missing check.
    """
    if not top_logprobs:
        return False
    return is_answer(max(top_logprobs, key=lambda pair: pair[1])[0])


# Step 1: pure scoring -- softmax(p(positive), p(negative)) over top-logprobs.
def positive_score(top_logprobs: list[tuple[str, float]]) -> float | None:
    """Return P(positive | {positive, negative}), or None if unanswered.

    Token surface forms are normalized (strip + lower) so leading-space and
    case variants from the tokenizer map onto the two canonical labels; the
    best (max) logprob per label wins when variants collide.

    None (a safe reject, per this module's contract) whenever the judge did not
    answer: no label came back, or the argmax is not a label.

    A label absent from the top-k is bounded, not assumed zero: it holds at most
    the smallest listed probability and at most the mass the list leaves
    unlisted. The score is therefore a lower bound when "positive" is the listed
    label and at most 0.5 when "negative" is.
    """
    best: dict[str, float] = {}
    for token, logprob in top_logprobs:
        label = token.strip().lower()
        if label in JUDGE_TOKENS:
            best[label] = max(best[label], logprob) if label in best else logprob
    if not best:
        return None
    if not _answered(top_logprobs, lambda token: token.strip().lower() in JUDGE_TOKENS):
        return None
    listed = [math.exp(logprob) for _, logprob in top_logprobs]
    absent_bound = min(min(listed), max(0.0, 1.0 - sum(listed)))
    pos = math.exp(best["positive"]) if "positive" in best else absent_bound
    neg = math.exp(best["negative"]) if "negative" in best else absent_bound
    return pos / (pos + neg)


def _is_digit_token(token: str) -> bool:
    label = token.strip()
    return len(label) == 1 and label.isdigit()


def digit_expectation_score(top_logprobs: list[tuple[str, float]]) -> float | None:
    """SpecReason's "average" score_method: expected value over 0-9 digits.

    Renormalizes over just the single-digit candidates, so non-numeric tokens
    are discarded rather than counted against the score. None (a safe reject)
    when the judge did not answer with a digit at all: with a single stray digit
    in the list the expectation collapses onto that digit exactly, regardless of
    how little probability it carried, which is the same accept-direction
    failure ``positive_score`` has.
    """
    best: dict[str, float] = {}
    for token, logprob in top_logprobs:
        if _is_digit_token(token):
            label = token.strip()
            best[label] = max(best[label], logprob) if label in best else logprob
    if not best:
        return None
    if not _answered(top_logprobs, _is_digit_token):
        return None
    weights = {digit: math.exp(logprob) for digit, logprob in best.items()}
    total = sum(weights.values())
    return sum(int(digit) * weight for digit, weight in weights.items()) / total


# The two verify methods differ only in how a line is asked about and how the
# single scored token is read back. Everything else -- the message layout, the
# hard gates, the accept/reject/fallback flow -- is shared.
JUDGE_METHODS = {
    "dreamr": (JUDGE_INSTRUCTION, positive_score),
    "sr": (SR_JUDGE_INSTRUCTION, digit_expectation_score),
}


# Step 2: threshold policy, boundary inclusive; missing tokens reject safely.
def verdict_from_top_logprobs(
    top_logprobs: list[tuple[str, float]],
    eta: float,
    verify_method: str = "dreamr",
) -> LineVerdict:
    score = JUDGE_METHODS[verify_method][1](top_logprobs)
    if score is None:
        return LineVerdict(accepted=False, score=None, reason="judge_tokens_missing")
    if score >= eta:
        return LineVerdict(accepted=True, score=score, reason="judge_accept")
    return LineVerdict(accepted=False, score=score, reason="judge_reject")


# Step 3: deterministic hard gates (free; run before the judge).
def hard_gate(line: str, accepted_lines: list[str], max_line_chars: int) -> tuple[bool, str]:
    """Return (ok, reason). Rejections: empty_line / duplicate_line / line_too_long."""
    stripped = line.strip()
    if not stripped:
        return False, "empty_line"
    if len(line) > max_line_chars:
        return False, "line_too_long"

    def normalize(text: str) -> str:
        return " ".join(text.lower().split())

    if normalize(stripped) in {normalize(prev) for prev in accepted_lines}:
        return False, "duplicate_line"
    return True, "ok"


# Step 4: the judge prompt -- full trajectory + slot spec + accepted + proposal.
def build_judge_messages(
    context_text: str,
    slot: str,
    accepted_lines: list[str],
    proposed_line: str,
    verify_method: str = "dreamr",
    prefix_messages: list[dict] | None = None,
) -> list[dict]:
    """Assemble the judge turn, optionally on top of the agent's own messages.

    Slot-specific text sits AFTER the trajectory, so all of a context's judge
    calls -- every slot, and every line within a slot -- share one prefix that
    the engine caches once.

    ``prefix_messages`` is the agent's real message list, for the case where
    the verify engine is the same server that just served the agent's turn
    (slurm/run_ours_e2e_variants.sbatch points --verify-llm-base-url at
    $TARGET_PORT). The history is already resident there, so sending the
    agent's messages verbatim and appending the judge turn hits that cache
    instead of prefilling a flattened re-render of the same conversation. It
    has to be the real list: the agent's prompt is many messages rendered
    through the chat template, and a single user turn holding the same text as
    a transcript diverges from it at the second token.

    When a prefix is supplied the trajectory is not repeated inside the judge
    turn -- it is already above, in its original form. Repeating it would
    make every judge call cost MORE to prefill than it does today.
    """
    instruction = JUDGE_METHODS[verify_method][0]
    accepted_block = "\n".join(accepted_lines) if accepted_lines else "(none yet)"
    trajectory_block = (
        ""
        if prefix_messages is not None
        else f"Trajectory:\n<trajectory>\n{context_text}\n</trajectory>\n\n"
    )
    content = (
        f"{instruction}\n\n"
        f"{trajectory_block}"
        f"Summary slot: {slot} -- {SLOT_CONTRACTS[slot]}\n\n"
        f"Accepted summary lines so far:\n{accepted_block}\n\n"
        f"Proposed next line:\n{proposed_line}"
    )
    return list(prefix_messages or []) + [{"role": "user", "content": content}]


# Step 5: the HTTP judge call, isolated for testability (fake servers in tests).
def judge_line(
    *,
    base_url: str,
    model: str,
    api_key: str,
    context_text: str,
    slot: str,
    accepted_lines: list[str],
    proposed_line: str,
    eta: float,
    timeout_seconds: int = 120,
    verify_method: str = "dreamr",
    prefix_messages: list[dict] | None = None,
    tools: list[dict] | None = None,
    usage_out: dict | None = None,
) -> LineVerdict:
    """One single-token judgment call against an OpenAI-compatible chat endpoint."""
    import requests

    payload = {
        "model": model,
        "messages": build_judge_messages(
            context_text,
            slot,
            accepted_lines,
            proposed_line,
            verify_method=verify_method,
            prefix_messages=prefix_messages,
        ),
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # Sent whenever a prefix is: a chat template that folds the tools schema
    # into the system message renders a different first message without them,
    # so the prefix would miss the cache from its very first token -- silently,
    # since the judge still returns a plausible score.
    if tools:
        payload["tools"] = tools
    from utils import measurement_clock as time

    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout_seconds,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    # A judge call reads back one token but prefills the whole prefix, so its
    # prompt_tokens is the only place that cost is visible; a caller summing
    # generation tokens alone understates verified compaction badly.
    if usage_out is not None:
        usage = body.get("usage") or {}
        usage_out["prompt_tokens"] = usage.get("prompt_tokens")
        usage_out["completion_tokens"] = usage.get("completion_tokens")
        # How much of that prefix the server already had cached. The judge
        # re-sends the same context on every line, so with APC on this is
        # nearly all of prompt_tokens and the call is far cheaper than the
        # raw count suggests; with APC off (any YaRN-served run) it is 0 and
        # every judge call prefills the whole thing again.
        usage_out["cached_prompt_tokens"] = (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )
        # max_tokens=1 above, so this is the whole call, not a prefill/decode
        # split: ttft_seconds here is the round trip and decode_seconds is 0 by
        # construction. Written unconditionally -- unlike the streamed draft
        # path there is no cheaper non-timing variant to fall back to, so
        # capture_timing does not gate it. Callers summing ttft_seconds across
        # roles are therefore mixing a real TTFT with a whole-call wall time;
        # see compaction_llm_timing_by_role in method.py for the split that
        # keeps them apart.
        usage_out["ttft_seconds"] = elapsed
        usage_out["decode_seconds"] = 0.0
    entries = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    top = [(entry["token"], entry["logprob"]) for entry in entries]
    return verdict_from_top_logprobs(top, eta, verify_method=verify_method)
