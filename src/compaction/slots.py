"""Spec-compact orchestration: line-level speculative loops over 6 slots.

Pure orchestration -- the draft generator, the judge, and the target fallback
are injected callables (HTTP clients bind them in the serving layer), so the
whole engine is testable offline with deterministic fakes:

    draft_fn(slot, accepted_lines)    -> (line | None, done: bool)
    judge_fn(slot, accepted_lines, line) -> LineVerdict
    fallback_fn(slot, accepted_lines) -> line (target-authored, trusted:
                                         hard-gated but never judged)

Per line: hard gate -> judge gate -> accept, or fallback from the break point
(the only target generation; standard SD semantics). A judge exception counts
as a rejection (server hiccups must not kill a compaction event).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from compaction.prompts import SLOT_TITLES, SLOTS
from compaction.verification import LineVerdict, hard_gate
from utils import measurement_clock as time


@dataclass
class SlotResult:
    slot: str
    lines: list[str]
    finished_reason: str  # draft_done | line_cap | fallback_rejected | token_budget
    stats: dict = field(default_factory=dict)
    # Per-line event stream for end-to-end inspection: each entry records the
    # proposed line, its gate outcome, and the action taken.
    events: list[dict] = field(default_factory=list)


# Step 1: the per-slot line-level speculative loop.
def compact_slot(
    *,
    slot: str,
    draft_fn: Callable[[str, list[str]], tuple[Optional[str], bool]],
    judge_fn: Optional[Callable[[str, list[str], str], "LineVerdict"]] = None,
    fallback_fn: Optional[Callable[[str, list[str]], str]] = None,
    max_lines: int,
    max_line_chars: int,
    fallback_executor: Optional["ThreadPoolExecutor"] = None,
    lookahead: int = 1,
    judge_batch_fn: Optional[
        Callable[[str, list[list[str]], list[str]], list["LineVerdict"]]
    ] = None,
    max_slot_tokens: int | None = None,
    token_count_fn: Callable[[str], int] | None = None,
) -> SlotResult:
    """``fallback_executor`` turns on DREAM-R's parallel schedule: the target's
    version of the next line is launched at the same time as the draft's,
    instead of only after a rejection. A rejection then no longer waits for a
    target round-trip it could have started earlier; an acceptance throws that
    work away. Both the break point and the prefix are known before the draft
    is asked, which is what makes the two safe to race.

    ``judge_fn``/``fallback_fn`` are optional together: omit both to run the
    draft's lines unverified, with no target round-trip at all -- every line
    that passes the deterministic hard gate is accepted as-is. A hard-gate
    failure still ends the slot (``finished_reason="fallback_rejected"``)
    rather than looping, since there is no target to repair it from.

    ``lookahead`` > 1 (with ``judge_batch_fn``) trades draft work for judge
    ROUND TRIPS. Measured on job 16546026 at concurrency=1, the judge cost
    18-22% of elapsed while generating exactly one token per call: its expense
    is not compute and not a cache miss (the server served 79-81% of its blocks
    from the prefix cache, with zero queue time) but a ~0.36-0.47s per-request
    floor paid ~13.5 times per row. k judgments issued together pay that floor
    once instead of k times.

    It cannot be done without speculating: line N+1's proposal is drafted from
    a prefix that includes line N, so N's verdict is not knowable before N+1
    exists. So the batch is drafted forward optimistically, judged in one shot,
    and then accepted as the LONGEST PASSING PREFIX -- at the first rejection
    the remaining proposals are discarded (counted as ``draft_discarded``) and
    the target repairs from the break point, exactly as in the k=1 loop. When
    every line passes, the accepted prefix is byte-identical to what k=1 would
    have produced from the same draft; the two differ only after a rejection,
    where k=1 would have re-drafted rather than discarded.

    The parallel (DREAM-R) fallback race is OFF under lookahead: a race started
    at the batch's break point only covers a rejection at position 0, and
    starting one per speculative position would issue the target work the batch
    exists to avoid.
    """
    if max_slot_tokens is not None:
        if max_slot_tokens < 1 or token_count_fn is None:
            raise ValueError("A positive slot token budget requires an exact token counter.")
        if lookahead != 1:
            raise ValueError("Token-bounded slot evaluation currently requires lookahead=1.")
    accepted: list[str] = []
    events: list[dict] = []
    stats = {
        "accepted": 0,
        "fallback": 0,
        "judge_rejects": 0,
        "judge_errors": 0,
        # Rejections because the judge did not answer the question (no usable
        # positive/negative or digit in its top-logprobs). Counted apart from
        # judge_rejects: those are the judge saying no, these are the judge
        # saying nothing, and a run where the second is common has a broken
        # judge prompt or template rather than a bad draft.
        "judge_tokens_missing": 0,
        "hard_rejects": {},
        "judge_scores": [],
        # Per-phase wall time (seconds) summed over this slot's calls. draft is
        # where parallelism x prefix-caching act; judge/fallback are target-side
        # and roughly constant across those knobs -- splitting them keeps the
        # draft-side signal from being swamped by ~N target round-trips.
        "draft_seconds": 0.0,
        "judge_seconds": 0.0,
        "fallback_seconds": 0.0,
        "slot_seconds": 0.0,
        # Round-trips actually paid for, per phase. Not derivable from the
        # outcome counters above: accepted + judge_rejects + hard_rejects misses
        # blank lines and the terminal `line is None` call, so "how many draft
        # calls did this slot cost" -- the denominator of any per-call latency
        # or tokens/second figure -- has no other source.
        "draft_calls": 0,
        "judge_calls": 0,
        "fallback_calls": 0,
        # Lookahead only. judge_batches counts ROUND TRIPS where judge_calls
        # counts judgments, so judge_calls/judge_batches is the batching factor
        # actually achieved -- always below `lookahead`, because a batch is cut
        # short by eos, the line cap, a blank line or a hard-gate failure.
        # draft_discarded is what the trade costs: lines drafted past a
        # rejection that were thrown away unjudged.
        "judge_batches": 0,
        "draft_discarded": 0,
    }
    slot_started = time.perf_counter()
    budget_exhausted = False
    if max_slot_tokens is not None:
        stats.update(
            {
                "max_slot_tokens": max_slot_tokens,
                "budget_discarded_draft": 0,
                "budget_discarded_target": 0,
            }
        )

    def _event(source: str, line: str, decision: str, reason, score=None, call=None) -> dict:
        """One event stream entry, stamped with the call that produced it.

        ``slot``/``call`` are the join key back to per-call token and timing
        records kept by the caller (method.py wraps the same three callables to
        collect ``usage_out``): the two clocks watch the same calls but neither
        can name the other's rows without it. The ordinal is that source's own
        call count, so a draft event's ``call`` indexes draft calls only.
        Exact under the sequential schedule; under the parallel one a raced
        fallback that is thrown away never reaches an event at all, so target
        ordinals there count only the races that were waited on.

        ``call`` overrides the counter, and the lookahead path MUST pass it:
        there the events are emitted in the acceptance pass, long after the
        whole batch has been drafted, so the live counter names the last call
        of the batch rather than the one that produced this line. Reading the
        counter there silently mis-joined every batched line to another line's
        token record -- the accepted text stayed correct, but the per-line and
        provenance tables built from the join did not.
        """
        if call is None:
            call = stats["fallback_calls"] if source == "target" else stats["draft_calls"]
        return {
            "slot": slot,
            "call": call,
            "source": source,
            "line": line,
            "decision": decision,
            "reason": reason,
            "score": score,
        }

    def fits_budget(line: str, source: str) -> bool:
        nonlocal budget_exhausted
        if max_slot_tokens is None:
            return True
        count = token_count_fn("\n".join([*accepted, line]))
        if count <= max_slot_tokens:
            return True
        budget_exhausted = True
        stats["budget_discarded_" + ("target" if source == "target" else "draft")] += 1
        event = _event(source, line, "budget_stop", "slot_token_budget")
        event["candidate_slot_tokens"] = count
        events.append(event)
        return False

    def try_fallback(pending=None) -> bool:
        """Target writes the line from the break point; hard-gated, not judged.

        ``pending`` is a Future already racing the draft; waiting on it here
        costs only whatever is left, which is the whole point of the parallel
        schedule. Unverified runs have no fallback_fn at all, so there is
        nothing to try -- the caller's gate failure ends the slot instead.
        """
        if fallback_fn is None:
            return False
        t0 = time.perf_counter()
        raw = pending.result() if pending is not None else fallback_fn(slot, accepted)
        stats["fallback_seconds"] += time.perf_counter() - t0
        stats["fallback_calls"] += 1
        # Same split as the draft path: the trailing newline is prompt state,
        # not content, and the gate must not see it.
        line = raw.rstrip("\n").rstrip()
        separator = raw[len(raw.rstrip("\n")) :]
        ok, reason = hard_gate(line, accepted, max_line_chars)
        if not ok:
            events.append(_event("target", line, "fallback_rejected", reason))
            return False
        if not fits_budget(line + separator, "target"):
            return False
        accepted.append(line + separator)
        stats["fallback"] += 1
        events.append(_event("target", line, "fallback_accepted", "ok"))
        return True

    stats["fallback_discarded"] = 0

    def split(raw: str) -> tuple[str, str]:
        """Content vs trailing-newline separator -- the k=1 loop's rule, shared.

        The separator is prompt-reconstruction state, not content: it must not
        reach the gate, the judge or the dedup check, but it has to survive
        into ``accepted`` because rejoining with "\\n" is what turns it back
        into the blank line the slot format calls for. rstrip only, since
        leading whitespace is meaningful for nested list items.
        """
        content = raw.rstrip("\n")
        separator = raw[len(content) :]
        return content.rstrip(), separator

    def run_batched() -> str:
        """The lookahead loop. Returns finished_reason; mutates accepted/stats."""
        while len(accepted) < max_lines:
            # --- speculate: draft forward from a prefix that assumes acceptance
            proposals: list[dict] = []
            speculative = list(accepted)
            stop_reason = None
            while len(proposals) < lookahead and len(speculative) < max_lines:
                t0 = time.perf_counter()
                raw, done = draft_fn(slot, speculative)
                stats["draft_seconds"] += time.perf_counter() - t0
                stats["draft_calls"] += 1
                if raw is None:
                    stop_reason = "draft_done"
                    break
                content, separator = split(raw)
                if not content:
                    # A blank line carries no claim, so it never costs a judge
                    # round trip -- same as k=1. It still has to sit in the
                    # speculative prefix, or the next draft call is asked to
                    # continue from a paragraph break that has not happened.
                    proposals.append(
                        {
                            "kind": "blank",
                            "sep": separator,
                            "done": done,
                            "call": stats["draft_calls"],
                        }
                    )
                    speculative.append(separator)
                    if done:
                        stop_reason = "draft_done"
                        break
                    continue
                ok, reason = hard_gate(content, speculative, max_line_chars)
                proposals.append(
                    {
                        "kind": "line",
                        "content": content,
                        "sep": separator,
                        "done": done,
                        "gate_ok": ok,
                        "gate_reason": reason,
                        # Stamped at DRAFT time -- see _event's `call` note.
                        "call": stats["draft_calls"],
                        # Snapshot: the judge for this line must see the prefix as
                        # it stood BEFORE the line, not the whole speculative run.
                        "prefix": list(speculative),
                    }
                )
                if not ok:
                    # Nothing past a hard-gate failure can be accepted, so
                    # drafting further would be pure waste.
                    break
                speculative.append(content + separator)
                if done:
                    stop_reason = "draft_done"
                    break

            if not proposals:
                return stop_reason or "line_cap"

            # --- judge every surviving proposal in ONE round trip
            judged = [p for p in proposals if p["kind"] == "line" and p["gate_ok"]]
            verdicts: list[LineVerdict] = []
            if judged:
                t0 = time.perf_counter()
                try:
                    verdicts = list(
                        judge_batch_fn(
                            slot, [p["prefix"] for p in judged], [p["content"] for p in judged]
                        )
                    )
                except Exception:
                    # Same contract as the k=1 loop: a judge that raises is a
                    # rejection, not a crash. The whole batch fails together
                    # because one transport failure took all of them down.
                    stats["judge_errors"] += len(judged)
                    verdicts = [
                        LineVerdict(accepted=False, score=None, reason="judge_error")
                        for _ in judged
                    ]
                stats["judge_seconds"] += time.perf_counter() - t0
                stats["judge_calls"] += len(judged)
                stats["judge_batches"] += 1
            by_id = dict(zip((id(p) for p in judged), verdicts))

            # --- accept the longest passing prefix
            for position, proposal in enumerate(proposals):
                if proposal["kind"] == "blank":
                    accepted.append(proposal["sep"])
                    events.append(_event("draft", "", "blank_line", "ok", call=proposal["call"]))
                    if proposal["done"]:
                        return "draft_done"
                    continue
                content = proposal["content"]
                if not proposal["gate_ok"]:
                    stats["hard_rejects"][proposal["gate_reason"]] = (
                        stats["hard_rejects"].get(proposal["gate_reason"], 0) + 1
                    )
                    events.append(
                        _event(
                            "draft",
                            content,
                            "hard_reject",
                            proposal["gate_reason"],
                            call=proposal["call"],
                        )
                    )
                    stats["draft_discarded"] += len(proposals) - position - 1
                    if not try_fallback():
                        return "fallback_rejected"
                    break
                verdict = by_id[id(proposal)]
                if verdict.score is not None:
                    stats["judge_scores"].append(verdict.score)
                if verdict.accepted:
                    accepted.append(content + proposal["sep"])
                    stats["accepted"] += 1
                    events.append(
                        _event(
                            "draft",
                            content,
                            "accepted",
                            verdict.reason,
                            verdict.score,
                            call=proposal["call"],
                        )
                    )
                    if proposal["done"]:
                        return "draft_done"
                    continue
                if verdict.reason == "judge_reject":
                    stats["judge_rejects"] += 1
                elif verdict.reason == "judge_tokens_missing":
                    stats["judge_tokens_missing"] += 1
                events.append(
                    _event(
                        "draft",
                        content,
                        "rejected",
                        verdict.reason,
                        verdict.score,
                        call=proposal["call"],
                    )
                )
                # Everything drafted after the break point was conditioned on
                # this line being accepted, so it is not merely unjudged -- it
                # is invalid. Discard and re-draft from the repaired prefix.
                stats["draft_discarded"] += len(proposals) - position - 1
                if not try_fallback():
                    return "fallback_rejected"
                break
            else:
                # Every proposal in the batch was consumed without a break, so
                # the reason the batch ended is the reason the draft stopped.
                if stop_reason == "draft_done":
                    return "draft_done"
        return "line_cap"

    if lookahead > 1 and judge_batch_fn is not None:
        finished_reason = run_batched()
        stats["slot_seconds"] = time.perf_counter() - slot_started
        return SlotResult(
            slot=slot,
            lines=accepted,
            finished_reason=finished_reason,
            stats=stats,
            events=events,
        )

    finished_reason = "line_cap"
    while len(accepted) < max_lines:
        if max_slot_tokens is not None and token_count_fn("\n".join(accepted)) >= max_slot_tokens:
            budget_exhausted = True
            finished_reason = "token_budget"
            break
        # Launched before the draft is asked, from the same prefix, so the two
        # genuinely overlap. Snapshot the prefix: `accepted` is mutated below
        # and the target must write from the break point as it stands now.
        racing = (
            fallback_executor.submit(fallback_fn, slot, list(accepted))
            if fallback_executor is not None and fallback_fn is not None
            else None
        )
        t0 = time.perf_counter()
        line, done = draft_fn(slot, accepted)
        stats["draft_seconds"] += time.perf_counter() - t0
        stats["draft_calls"] += 1
        if line is None:
            finished_reason = "draft_done"
            break
        # done=True with a line means the draft emitted eos right after this
        # final line: gate it like any other, then end the slot.
        #
        # A trailing newline is the client's record that the draft asked for a
        # blank line after this one (spec_clients.DraftLineClient). Split it
        # off: it is prompt-reconstruction state, not content, so it must not
        # reach the gate, the judge, or the dedup check -- but it must survive
        # into `accepted`, because rejoining with "\n" is what turns it back
        # into the blank line the slot's format calls for. rstrip only, since
        # leading whitespace is meaningful for nested list items.
        content = line.rstrip("\n")
        separator = line[len(content) :]
        content = content.rstrip()
        if not fits_budget(content + separator, "draft"):
            if racing is not None:
                stats["fallback_discarded"] += 1
            finished_reason = "token_budget"
            break

        if not content:
            # The draft asked for a blank line and nothing else. There is no
            # claim to verify, so it does not get a judge round-trip; keeping
            # it preserves the block structure the slot format requires.
            accepted.append(separator)
            if racing is not None:
                stats["fallback_discarded"] += 1
            events.append(_event("draft", "", "blank_line", "ok"))
            if done:
                finished_reason = "draft_done"
                break
            continue

        line = content + separator

        # Step 1a: deterministic hard gate (free, before the judge).
        ok, reason = hard_gate(content, accepted, max_line_chars)
        if not ok:
            stats["hard_rejects"][reason] = stats["hard_rejects"].get(reason, 0) + 1
            events.append(_event("draft", content, "hard_reject", reason))
            if not try_fallback(racing):
                finished_reason = "fallback_rejected"
                break
            if done:
                finished_reason = "draft_done"
                break
            continue

        # Step 1b: the DREAM-R judge gate; exceptions count as rejections.
        # Skipped entirely when unverified -- a hard-gate pass is the only
        # gate, and the line is accepted straight from the draft.
        if judge_fn is None:
            accepted.append(line)
            stats["accepted"] += 1
            events.append(_event("draft", content, "accepted", "no_verify"))
            if done:
                finished_reason = "draft_done"
                break
            continue

        t0 = time.perf_counter()
        try:
            verdict = judge_fn(slot, accepted, content)
        except Exception:
            stats["judge_errors"] += 1
            verdict = LineVerdict(accepted=False, score=None, reason="judge_error")
        stats["judge_seconds"] += time.perf_counter() - t0
        stats["judge_calls"] += 1
        if verdict.score is not None:
            stats["judge_scores"].append(verdict.score)

        if verdict.accepted:
            accepted.append(line)
            stats["accepted"] += 1
            # The raced target line is thrown away. Counted, because it is the
            # cost of the parallel schedule: at a high accept rate almost every
            # one of these is wasted target work, and the trade only looks good
            # if that is on the table next to the latency it bought.
            if racing is not None:
                stats["fallback_discarded"] += 1
            events.append(_event("draft", content, "accepted", verdict.reason, verdict.score))
        else:
            if verdict.reason == "judge_reject":
                stats["judge_rejects"] += 1
            elif verdict.reason == "judge_tokens_missing":
                stats["judge_tokens_missing"] += 1
            events.append(_event("draft", content, "rejected", verdict.reason, verdict.score))
            if not try_fallback(racing):
                finished_reason = "fallback_rejected"
                break
        if done:
            finished_reason = "draft_done"
            break

    if budget_exhausted:
        finished_reason = "token_budget"
    if max_slot_tokens is not None:
        stats["output_slot_tokens"] = token_count_fn("\n".join(accepted))
    stats["slot_seconds"] = time.perf_counter() - slot_started
    return SlotResult(
        slot=slot, lines=accepted, finished_reason=finished_reason, stats=stats, events=events
    )


# Step 2: run all six slots concurrently and assemble the method.py-style summary.
def spec_compact(
    *,
    draft_fn,
    judge_fn=None,
    fallback_fn=None,
    max_lines: int,
    max_line_chars: int,
    max_workers: int = 6,
    verify_schedule: str = "sequential",
    lookahead: int = 1,
    judge_batch_fn=None,
    max_slot_tokens: int | None = None,
    token_count_fn: Callable[[str], int] | None = None,
) -> tuple[dict[str, SlotResult], str]:
    """``verify_schedule="parallel"`` races the target's line against the
    draft's (DREAM-R). It needs a POOL OF ITS OWN: with one shared pool every
    worker ends up holding a slot that is blocked on a fallback queued behind
    the slots still running, and nothing can finish. That deadlock was already
    paid for once on the slot-level path (c2799f6); the two-pool split makes it
    structurally impossible rather than merely unlikely.
    """
    batched = lookahead > 1 and judge_batch_fn is not None
    # See compact_slot's docstring: a race from the batch's break point only
    # covers a rejection at position 0, so under lookahead the parallel
    # schedule buys almost nothing and costs a target generation per batch.
    parallel = verify_schedule == "parallel" and not batched
    fallback_pool = (
        ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="spec-fallback")
        if parallel
        else None
    )

    def run(slot: str) -> SlotResult:
        return compact_slot(
            slot=slot,
            draft_fn=draft_fn,
            judge_fn=judge_fn,
            fallback_fn=fallback_fn,
            max_lines=max_lines,
            max_line_chars=max_line_chars,
            fallback_executor=fallback_pool,
            lookahead=lookahead,
            judge_batch_fn=judge_batch_fn,
            max_slot_tokens=max_slot_tokens,
            token_count_fn=token_count_fn,
        )

    try:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="spec-slot") as pool:
            results = dict(zip(SLOTS, pool.map(run, SLOTS)))
    finally:
        if fallback_pool is not None:
            # cancel_futures drops races still queued behind an accepted line;
            # they are pure waste and would keep loading the target long after
            # the compaction that spawned them returned.
            fallback_pool.shutdown(wait=False, cancel_futures=True)

    summary = "\n\n".join(
        f"## {SLOT_TITLES[slot]}\n" + "\n".join(results[slot].lines) for slot in SLOTS
    )
    return results, summary
