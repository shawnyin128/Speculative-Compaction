"""Compose the draft, semantic verifier and target repair over six parallel slots."""

import os
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from compaction.clients import DraftLineClient, TargetFallbackClient
from compaction.prompts import SLOTS, build_shared_prefix
from compaction.slots import spec_compact
from compaction.verification import LineVerdict, judge_line
from hparam.hparam import Compaction
from utils import measurement_clock as time

TOKENIZER_LOCK = threading.Lock()


def load_tokenizer(name):
    """Acquire an optional model dependency only for actual tokenization work."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


@dataclass
class CompactionResult:
    text: str
    metadata: dict[str, Any]


class Compactor:
    """Own configuration and tokenizer lifetimes; each call owns its slot clients.

    Args:
        parameters: Validated compaction settings. No context is truncated.
        draft_tokenizer: Optional already-loaded draft tokenizer.
        target_tokenizer: Optional matching target tokenizer for repair/budgets.
    """

    def __init__(self, parameters: Compaction, *, draft_tokenizer=None, target_tokenizer=None):
        parameters.validate()
        self.parameters = parameters
        self.draft_tokenizer = draft_tokenizer
        self.target_tokenizer = target_tokenizer

    def compact(self, context_text: str) -> CompactionResult:
        """Return committed slot text and attributed generation/acceptance counters.

        Input is an already-rendered context string. Over-budget prompts raise;
        the method neither selects historical turns nor truncates supplied text.
        Tokenizer loading is outside the measured compaction execution interval.
        """
        if not isinstance(context_text, str) or not context_text.strip():
            raise ValueError("context_text must be a nonempty string.")
        p = self.parameters
        # Step 1: load each model's own tokenizer before starting slot threads.
        with TOKENIZER_LOCK:
            if self.draft_tokenizer is None or self.target_tokenizer is None:
                if self.draft_tokenizer is None:
                    self.draft_tokenizer = load_tokenizer(p.draft.tokenizer)
                target_name = p.draft.tokenizer
                if p.target is not None:
                    target_name = p.target.tokenizer
                if self.target_tokenizer is None:
                    if target_name == p.draft.tokenizer:
                        self.target_tokenizer = self.draft_tokenizer
                    elif target_name != p.draft.tokenizer:
                        self.target_tokenizer = load_tokenizer(target_name)
        draft_client = DraftLineClient(
            base_url=p.draft.url.rstrip("/"),
            model=p.draft.model,
            api_key=os.environ.get(p.draft.api_key_env, "EMPTY"),
            context_text=context_text,
            tokenizer=self.draft_tokenizer,
            max_prompt_tokens=p.max_prompt_tokens,
            max_line_tokens=p.max_line_tokens,
            full_schema_prefix=p.full_schema_prefix,
            temperature=p.temperature,
            top_p=p.top_p,
            line_split=p.line_split,
            buffer_extra_lines=p.buffer_extra_lines,
            capture_timing=p.capture_timing,
            timeout_seconds=p.timeout_seconds,
        )
        fallback_client = None
        if p.verify:
            fallback_client = TargetFallbackClient(
                base_url=p.target.url.rstrip("/"),
                model=p.target.model,
                api_key=os.environ.get(p.target.api_key_env, "EMPTY"),
                context_text=context_text,
                tokenizer=self.target_tokenizer,
                max_prompt_tokens=p.max_prompt_tokens,
                max_line_tokens=p.max_line_tokens,
                temperature=0.0,
                top_p=None,
                buffer_extra_lines=False,
                full_schema_prefix=p.fallback_full_schema_prefix,
                line_split=p.fallback_line_split,
                capture_timing=p.capture_timing,
                timeout_seconds=p.timeout_seconds,
            )
        judge_prefix = None
        if p.verify_shared_prefix:
            judge_prefix = [
                {"role": "user", "content": build_shared_prefix(context_text, slots=SLOTS)}
            ]

        # Step 2: attribute all actual calls, including rejected/discarded work.
        by_role = defaultdict(
            lambda: {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_prompt_tokens": 0,
                "ttft_seconds": 0.0,
                "decode_seconds": 0.0,
                "wall_seconds": 0.0,
            }
        )
        ordinals = Counter()
        calls = []
        accounting_lock = threading.Lock()

        def measured(role, slot, usage, operation):
            started = time.perf_counter()
            usage["started_monotonic_seconds"] = started
            usage["started_unix_seconds"] = time.time()
            try:
                return operation()
            finally:
                finished = time.perf_counter()
                usage.update(
                    wall_seconds=finished - started,
                    finished_monotonic_seconds=finished,
                    finished_unix_seconds=time.time(),
                )
                with accounting_lock:
                    entry = by_role[role]
                    entry["calls"] += 1
                    for key in [
                        "prompt_tokens",
                        "completion_tokens",
                        "cached_prompt_tokens",
                        "ttft_seconds",
                        "decode_seconds",
                        "wall_seconds",
                    ]:
                        if usage.get(key) is not None:
                            entry[key] += type(entry[key])(usage[key])
                    if p.record_llm_calls:
                        ordinals[role, slot] += 1
                        calls.append(
                            {"role": role, "slot": slot, "call": ordinals[role, slot], **usage}
                        )

        def draft(slot, accepted):
            usage = {}
            return measured(
                "draft",
                slot,
                usage,
                lambda: draft_client.next_line(slot, accepted, usage_out=usage),
            )

        def repair(slot, accepted):
            usage = {}
            return measured(
                "fallback",
                slot,
                usage,
                lambda: fallback_client.fallback_line(slot, accepted, usage_out=usage),
            )

        def verify(slot, accepted, line):
            if p.structural_bypass:
                stripped = line.strip().lower()
                if stripped.startswith("#") or stripped.strip("-* ") in {"(none)", "none", "n/a"}:
                    return LineVerdict(accepted=True, score=None, reason="structural_bypass")
            usage = {}
            return measured(
                "judge",
                slot,
                usage,
                lambda: judge_line(
                    base_url=p.target.url.rstrip("/"),
                    model=p.target.model,
                    api_key=os.environ.get(p.target.api_key_env, "EMPTY"),
                    context_text=context_text,
                    slot=slot,
                    accepted_lines=accepted,
                    proposed_line=line,
                    eta=p.eta,
                    verify_method=p.verify_method,
                    prefix_messages=judge_prefix,
                    usage_out=usage,
                    timeout_seconds=p.timeout_seconds,
                ),
            )

        def verify_batch(slot, prefixes, lines):
            if len(lines) == 1:
                return [verify(slot, prefixes[0], lines[0])]
            with ThreadPoolExecutor(
                max_workers=len(lines), thread_name_prefix="spec-judge"
            ) as pool:
                return list(
                    pool.map(lambda pair: verify(slot, pair[0], pair[1]), zip(prefixes, lines))
                )

        # Step 3: preserve line ordering, acceptance/repair and slot budgets.
        judge_fn = None
        fallback_fn = None
        batch_fn = None
        if p.verify:
            judge_fn, fallback_fn, batch_fn = verify, repair, verify_batch
        count_tokens = None
        if p.max_slot_tokens is not None:

            def count_tokens(text):
                return len(self.target_tokenizer(text, add_special_tokens=False)["input_ids"])

        started = time.monotonic()
        results, text = spec_compact(
            draft_fn=draft,
            judge_fn=judge_fn,
            fallback_fn=fallback_fn,
            max_lines=p.max_lines,
            max_line_chars=p.max_line_chars,
            max_workers=p.slot_concurrency,
            verify_schedule=p.verify_schedule,
            lookahead=p.verify_lookahead,
            judge_batch_fn=batch_fn,
            max_slot_tokens=p.max_slot_tokens,
            token_count_fn=count_tokens,
        )
        summary_budget_tokens = None
        if p.summary_token_budget is not None:
            summary_budget_tokens = len(
                self.target_tokenizer(text, add_special_tokens=False)["input_ids"]
            )
            if summary_budget_tokens > p.summary_token_budget:
                raise ValueError(
                    f"Assembled summary exceeds its token budget: {summary_budget_tokens} > {p.summary_token_budget}."
                )
        elapsed = time.monotonic() - started

        # Step 4: summary size and engine work are different token denominators.
        slot_stats = {}
        for slot, result in results.items():
            scores = result.stats["judge_scores"]
            mean_score = None
            if scores:
                mean_score = sum(scores) / len(scores)
            slot_stats[slot] = {
                "lines": len(result.lines),
                "tokens": len(
                    self.draft_tokenizer("\n".join(result.lines), add_special_tokens=False)[
                        "input_ids"
                    ]
                ),
                "finished_reason": result.finished_reason,
                **{key: value for key, value in result.stats.items() if key != "judge_scores"},
                "structural_bypass_lines": sum(
                    event.get("reason") == "structural_bypass" for event in result.events
                ),
                "judge_score_mean": mean_score,
            }
        accepted = sum(stats["accepted"] for stats in slot_stats.values())
        repaired = sum(stats["fallback"] for stats in slot_stats.values())
        accept_rate = None
        if accepted + repaired:
            accept_rate = accepted / (accepted + repaired)
        metadata = {
            "method_name": "spec-compact",
            "eta": p.eta,
            "slot_concurrency": p.slot_concurrency,
            "trace_tokens": len(
                self.draft_tokenizer(context_text, add_special_tokens=False)["input_ids"]
            ),
            "compact_tokens": len(
                self.draft_tokenizer(text, add_special_tokens=False)["input_ids"]
            ),
            "summary_budget_tokens": summary_budget_tokens,
            "max_slot_tokens": p.max_slot_tokens,
            "summary_token_budget": p.summary_token_budget,
            "accepted_lines": accepted,
            "fallback_lines": repaired,
            "accept_rate": accept_rate,
            "elapsed_seconds": elapsed,
            "compact_seconds": round(elapsed, 3),
            "measurement_clock": time.CLOCK_SOURCE,
            "slot_stats": slot_stats,
            "output_chars": len(text),
            "compaction_llm_tokens": {
                "prompt": sum(v["prompt_tokens"] for v in by_role.values()),
                "completion": sum(v["completion_tokens"] for v in by_role.values()),
            },
            "compaction_llm_tokens_by_role": {
                role: {
                    key: entry[key]
                    for key in [
                        "calls",
                        "prompt_tokens",
                        "completion_tokens",
                        "cached_prompt_tokens",
                    ]
                }
                for role, entry in by_role.items()
            },
            "compaction_llm_timing_by_role": {
                role: {
                    key: round(entry[key], 4)
                    for key in ["ttft_seconds", "decode_seconds", "wall_seconds"]
                }
                for role, entry in by_role.items()
            },
        }
        if p.record_llm_calls:
            metadata["compaction_llm_calls"] = calls
            metadata["slot_events"] = {slot: result.events for slot, result in results.items()}
        return CompactionResult(text=text, metadata=metadata)
