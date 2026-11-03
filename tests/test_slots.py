"""Tests for the spec-compact orchestration engine (line loop + concurrency)."""

import pytest

from compaction.prompts import SLOTS
from compaction.slots import SlotResult, compact_slot, spec_compact
from compaction.verification import LineVerdict


def accept_all(slot, accepted, line):
    return LineVerdict(accepted=True, score=0.99, reason="judge_accept")


def scripted_draft(lines):
    state = {"i": 0}

    def draft_fn(slot, accepted):
        if state["i"] >= len(lines):
            return None, True
        line = lines[state["i"]]
        state["i"] += 1
        return line, False

    return draft_fn


def test_all_accept_flow_preserves_order_and_stats():
    draft = scripted_draft(["- a.", "- b.", "- c."])
    result = compact_slot(
        slot="goal",
        draft_fn=draft,
        judge_fn=accept_all,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=10,
        max_line_chars=400,
    )
    assert isinstance(result, SlotResult)
    assert result.lines == ["- a.", "- b.", "- c."]
    assert result.finished_reason == "draft_done"
    assert result.stats["accepted"] == 3
    assert result.stats["fallback"] == 0


def test_per_phase_timing_is_recorded():
    """The ablation needs draft/judge/fallback split out from total wall time so
    the draft-side (parallelism x caching) signal isn't swamped by target calls."""
    import time

    def slow_draft_fn(slot, accepted):
        if len(accepted) >= 2:
            return None, True
        time.sleep(0.01)
        return f"- line {len(accepted)}.", False

    def slow_judge(slot, accepted, line):
        time.sleep(0.01)
        return LineVerdict(accepted=True, score=0.9, reason="judge_accept")

    result = compact_slot(
        slot="goal",
        draft_fn=slow_draft_fn,
        judge_fn=slow_judge,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=10,
        max_line_chars=400,
    )
    for key in ("draft_seconds", "judge_seconds", "fallback_seconds", "slot_seconds"):
        assert key in result.stats and result.stats[key] >= 0.0
    # both phases ran twice -> each accumulated real time; slot wall covers both
    assert result.stats["draft_seconds"] > 0.0
    assert result.stats["judge_seconds"] > 0.0
    assert result.stats["slot_seconds"] >= result.stats["draft_seconds"]


def test_judge_reject_triggers_fallback_line():
    def judge(slot, accepted, line):
        bad = line == "- b."
        return LineVerdict(
            accepted=not bad,
            score=0.1 if bad else 0.9,
            reason="judge_reject" if bad else "judge_accept",
        )

    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(["- a.", "- b.", "- c."]),
        judge_fn=judge,
        fallback_fn=lambda s, a: "- repaired.",
        max_lines=10,
        max_line_chars=400,
    )
    assert result.lines == ["- a.", "- repaired.", "- c."]
    assert result.stats["fallback"] == 1
    assert result.stats["judge_rejects"] == 1


def test_hard_gate_duplicate_goes_to_fallback_without_judge():
    calls = {"judge": 0}

    def judge(slot, accepted, line):
        calls["judge"] += 1
        return LineVerdict(accepted=True, score=0.9, reason="judge_accept")

    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(["- a.", "- A.", "- c."]),
        judge_fn=judge,
        fallback_fn=lambda s, a: "- other.",
        max_lines=10,
        max_line_chars=400,
    )
    assert result.lines == ["- a.", "- other.", "- c."]
    assert result.stats["hard_rejects"]["duplicate_line"] == 1
    assert calls["judge"] == 2  # duplicate line never reaches the judge


def test_fallback_line_failing_hard_gate_stops_slot():
    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(["- a.", "- a."]),
        judge_fn=accept_all,
        fallback_fn=lambda s, a: "- a.",  # duplicate again
        max_lines=10,
        max_line_chars=400,
    )
    assert result.lines == ["- a."]
    assert result.finished_reason == "fallback_rejected"


def test_line_cap_stops_slot():
    def endless(slot, accepted):
        return "- line %d." % len(accepted), False

    result = compact_slot(
        slot="goal",
        draft_fn=endless,
        judge_fn=accept_all,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=3,
        max_line_chars=400,
    )
    assert len(result.lines) == 3
    assert result.finished_reason == "line_cap"


def test_judge_exception_counts_as_reject_and_falls_back():
    def judge(slot, accepted, line):
        raise RuntimeError("judge server down")

    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(["- a."]),
        judge_fn=judge,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=10,
        max_line_chars=400,
    )
    assert result.lines == ["- fb."]
    assert result.stats["judge_errors"] == 1
    assert result.finished_reason == "draft_done"


def test_spec_compact_runs_slots_concurrently_without_cross_talk():
    def make_draft(slot_name):
        return scripted_draft([f"- {slot_name} line 1.", f"- {slot_name} line 2."])

    drafts = {}

    def draft_fn(slot, accepted):
        if slot not in drafts:
            drafts[slot] = make_draft(slot)
        return drafts[slot](slot, accepted)

    results, summary = spec_compact(
        draft_fn=draft_fn,
        judge_fn=accept_all,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=10,
        max_line_chars=400,
        max_workers=6,
    )
    assert len(results) == 6
    for slot, res in results.items():
        assert res.lines == [f"- {slot} line 1.", f"- {slot} line 2."]
    # Assembly follows canonical slot order with method.py-style headings.
    positions = [summary.index(f"- {slot} line 1.") for slot in results]
    assert positions == sorted(positions)
    assert "## " in summary


def test_event_stream_records_every_decision_in_order():
    def judge(slot, accepted, line):
        bad = line == "- b."
        return LineVerdict(
            accepted=not bad,
            score=0.2 if bad else 0.9,
            reason="judge_reject" if bad else "judge_accept",
        )

    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(["- a.", "- b."]),
        judge_fn=judge,
        fallback_fn=lambda s, a: "- repaired.",
        max_lines=10,
        max_line_chars=400,
    )
    decisions = [(e["source"], e["decision"]) for e in result.events]
    assert decisions == [
        ("draft", "accepted"),
        ("draft", "rejected"),
        ("target", "fallback_accepted"),
    ]
    assert result.events[0]["score"] == 0.9
    assert result.events[1]["reason"] == "judge_reject"
    assert result.events[2]["line"] == "- repaired."


def test_final_line_with_done_flag_is_gated_and_kept():
    """done=True alongside a line (draft emitted eos) still gates the line."""

    def draft_fn(slot, accepted):
        return ("- final.", True) if not accepted else (None, True)

    result = compact_slot(
        slot="goal",
        draft_fn=draft_fn,
        judge_fn=accept_all,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=10,
        max_line_chars=400,
    )
    assert result.lines == ["- final."]
    assert result.finished_reason == "draft_done"
    assert result.stats["accepted"] == 1


# --- block structure ------------------------------------------------------
#
# The `progress` slot's trained format has blank lines between its ### Done /
# ### In Progress / ### Blocked sections -- 250 of 250 sampled SFT targets
# contain one. Whether those survive the line loop is invisible from the
# outside: a slot that loses them still returns plausible lines, and still
# reports finished_reason="draft_done".


def _scripted_draft(script):
    """draft_fn over a fixed list of (line, done) the client would return."""
    state = {"i": 0}

    def draft_fn(slot, accepted):
        if state["i"] >= len(script):
            return None, True
        out = script[state["i"]]
        state["i"] += 1
        return out

    return draft_fn


def _accept_all(recorded):
    def judge_fn(slot, accepted, line):
        recorded.append(line)
        return LineVerdict(accepted=True, score=1.0, reason="judge_accept")

    return judge_fn


def test_paragraph_breaks_survive_into_the_assembled_slot():
    """A blank line between blocks must reach the output, unjudged."""
    judged = []
    # "- [x] b\n" is the client's record that the draft asked for a blank line
    # after that bullet (spec_clients.DraftLineClient).
    result = compact_slot(
        slot="progress",
        draft_fn=_scripted_draft(
            [
                ("### Done", False),
                ("- [x] b\n", False),
                ("### In Progress", False),
                ("- [ ] c", True),
            ]
        ),
        judge_fn=_accept_all(judged),
        fallback_fn=lambda s, a: "",
        max_lines=12,
        max_line_chars=400,
    )

    assert result.finished_reason == "draft_done"
    assert "\n".join(result.lines) == ("### Done\n- [x] b\n\n### In Progress\n- [ ] c")
    # The marker is prompt state, not a claim -- the judge sees the bullet, not
    # the bullet plus a newline.
    assert judged == ["### Done", "- [x] b", "### In Progress", "- [ ] c"]


def test_a_blank_line_alone_costs_no_judge_call():
    """Nothing to verify in an empty line, so it must not spend a round-trip.

    Judge round-trips dominate compaction wall-clock, so a blank line that
    silently bought one would be a real cost, not a cosmetic one.
    """
    judged = []
    result = compact_slot(
        slot="progress",
        draft_fn=_scripted_draft([("### Done", False), ("\n", False), ("### Blocked", True)]),
        judge_fn=_accept_all(judged),
        fallback_fn=lambda s, a: "",
        max_lines=12,
        max_line_chars=400,
    )
    assert judged == ["### Done", "### Blocked"]
    assert any(e["decision"] == "blank_line" for e in result.events)


# --- the ablation axes the line path exposes ------------------------------
#
# These were unreachable: spec_compact hardcoded max_workers=6 and nothing set
# DraftLineClient.full_schema_prefix, so the serial arm and the shared-prefix
# arm of the draft-side ablation could not be run at line granularity at all.


# --- parallel fallback schedule -------------------------------------------


class _AlwaysReject:
    def __call__(self, slot, accepted, line):
        return LineVerdict(accepted=False, score=0.0, reason="judge_reject")


@pytest.mark.parametrize("workers", [1, 2, 3, 6])
def test_parallel_schedule_never_deadlocks_on_reject(workers):
    """Every line rejecting, at every pool size, must still finish.

    A single shared pool deadlocks here: all six slot loops are submitted at
    once, so once each worker holds a slot blocked on its raced fallback, the
    queued fallbacks can never start. The slot-level path already paid for this
    (c2799f6) -- concurrency 2 and 3 hung while 1 and >=6 survived by accident,
    so a test at only one pool size proves nothing.
    """
    from compaction.slots import spec_compact

    lines = iter(lambda: ("- draft line", False), None)
    results, _ = spec_compact(
        draft_fn=lambda slot, accepted: next(lines),
        judge_fn=_AlwaysReject(),
        fallback_fn=lambda slot, accepted: "- target line",
        max_lines=3,
        max_line_chars=400,
        max_workers=workers,
        verify_schedule="parallel",
    )
    assert set(results) == set(SLOTS)
    assert all(r.stats["fallback"] > 0 for r in results.values())


def test_parallel_schedule_counts_the_work_it_throws_away():
    """An accepted line discards its raced target generation; that is the
    schedule's whole cost and has to be visible next to the latency it buys."""
    from compaction.slots import spec_compact

    results, _ = spec_compact(
        draft_fn=_scripted_draft([("- a", False), ("- b", True)]),
        judge_fn=_accept_all([]),
        fallback_fn=lambda slot, accepted: "- target line",
        max_lines=4,
        max_line_chars=400,
        max_workers=2,
        verify_schedule="parallel",
    )
    assert sum(r.stats["fallback_discarded"] for r in results.values()) > 0


def test_sequential_schedule_launches_no_speculative_fallback():
    calls = []
    from compaction.slots import spec_compact

    spec_compact(
        draft_fn=_scripted_draft([("- a", True)]),
        judge_fn=_accept_all([]),
        fallback_fn=lambda slot, accepted: calls.append(slot) or "",
        max_lines=4,
        max_line_chars=400,
        max_workers=2,
    )
    assert calls == []


def test_unverified_accepts_every_line_that_passes_the_hard_gate():
    """judge_fn=None, fallback_fn=None: every hard-gate-passing line is kept
    straight from the draft, with no target round-trip at all."""
    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(["- a.", "- b.", "- c."]),
        max_lines=10,
        max_line_chars=400,
    )
    assert result.lines == ["- a.", "- b.", "- c."]
    assert result.finished_reason == "draft_done"
    assert result.stats["accepted"] == 3
    assert result.stats["fallback"] == 0
    assert result.stats["judge_errors"] == 0
    assert all(
        event["reason"] == "no_verify"
        for event in result.events
        if event["source"] == "draft" and event["decision"] == "accepted"
    )


def test_unverified_hard_gate_failure_ends_the_slot_without_a_fallback_call():
    """No fallback_fn to try, so a duplicate line ends the slot rather than
    looping or crashing on a None call."""
    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(["- a.", "- A.", "- c."]),  # "- A." duplicates "- a."
        max_lines=10,
        max_line_chars=400,
    )
    assert result.lines == ["- a."]
    assert result.finished_reason == "fallback_rejected"
    assert result.stats["hard_rejects"]["duplicate_line"] == 1


def test_unverified_ignores_parallel_schedule_without_a_fallback_pool_crash():
    """verify_schedule='parallel' with no fallback_fn must not try to submit
    None to the racing executor. draft_fn is keyed off `accepted` (per-slot
    state, not a shared iterator) so every slot's result is deterministic
    even though all six run concurrently against the same callable."""
    from compaction.slots import spec_compact

    def one_line_then_done(slot, accepted):
        if accepted:
            return None, True
        return "- a", False

    results, text = spec_compact(
        draft_fn=one_line_then_done,
        max_lines=4,
        max_line_chars=400,
        max_workers=2,
        verify_schedule="parallel",
    )
    assert set(results) == set(SLOTS)
    assert all(r.lines == ["- a"] for r in results.values())
    assert "## Goal" in text


def test_call_counts_and_event_ordinals_are_recorded():
    """Round-trips paid for, and which call produced which line.

    Neither is derivable from the outcome counters: accepted + judge_rejects +
    hard_rejects misses blank lines and the terminal empty draft call, and the
    event stream had no way to name the call it came from -- which is the join
    key a per-call token record needs to be split by accept/reject.
    """

    def judge(slot, accepted, line):
        bad = line == "- b."
        return LineVerdict(
            accepted=not bad,
            score=0.2 if bad else 0.9,
            reason="judge_reject" if bad else "judge_accept",
        )

    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(["- a.", "- b."]),
        judge_fn=judge,
        fallback_fn=lambda s, a: "- repaired.",
        max_lines=10,
        max_line_chars=400,
    )
    # Three draft calls: two lines plus the one that returned None and ended
    # the slot. That third call is real latency and is invisible in `accepted`.
    assert result.stats["draft_calls"] == 3
    assert result.stats["judge_calls"] == 2
    assert result.stats["fallback_calls"] == 1
    assert [(e["source"], e["call"]) for e in result.events] == [
        ("draft", 1),
        ("draft", 2),
        ("target", 1),
    ]
    assert {e["slot"] for e in result.events} == {"goal"}


# --- lookahead: batched judging -------------------------------------------


def batch_of(verdicts):
    """A judge_batch_fn returning scripted verdicts, recording each batch size."""
    calls = []

    def judge_batch_fn(slot, prefixes, lines):
        assert len(prefixes) == len(lines)
        calls.append(list(lines))
        return [verdicts[line] for line in lines]

    judge_batch_fn.calls = calls
    return judge_batch_fn


def test_lookahead_judges_a_whole_batch_in_one_round_trip():
    """The point of the feature: k judgments, one round trip, same accepted text.

    With every line accepted the result must be byte-identical to the k=1 loop
    (test_all_accept_flow_preserves_order_and_stats above asserts the same
    lines), so batching is observable only in judge_batches.
    """
    lines = ["- a.", "- b.", "- c."]
    batch = batch_of({line: LineVerdict(True, 0.99, "judge_accept") for line in lines})
    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(lines),
        judge_fn=accept_all,
        judge_batch_fn=batch,
        lookahead=4,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=10,
        max_line_chars=400,
    )
    assert result.lines == lines
    assert result.finished_reason == "draft_done"
    assert result.stats["accepted"] == 3
    assert result.stats["judge_batches"] == 1, "3 lines must cost ONE round trip"
    assert result.stats["judge_calls"] == 3
    assert result.stats["draft_discarded"] == 0
    assert batch.calls == [lines]


def test_lookahead_accepts_only_the_prefix_before_a_rejection():
    """A rejection mid-batch invalidates everything drafted after it.

    Lines after the break point were drafted from a prefix that assumed the
    rejected line was accepted, so keeping them would splice text onto a
    history that never happened -- they must be discarded and counted, and the
    target's repair line takes the break point instead.
    """
    lines = ["- a.", "- bad.", "- c."]
    batch = batch_of(
        {
            "- a.": LineVerdict(True, 0.99, "judge_accept"),
            "- bad.": LineVerdict(False, 0.01, "judge_reject"),
            "- c.": LineVerdict(True, 0.99, "judge_accept"),
        }
    )
    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(lines),
        judge_fn=accept_all,
        judge_batch_fn=batch,
        lookahead=3,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=10,
        max_line_chars=400,
    )
    assert result.lines[:2] == ["- a.", "- fb."], "repair replaces the rejected line"
    assert "- c." not in result.lines, "speculation past the break point is invalid"
    assert result.stats["judge_rejects"] == 1
    assert result.stats["fallback"] == 1
    assert result.stats["draft_discarded"] == 1


def test_lookahead_judge_failure_rejects_the_batch_without_crashing():
    """Same contract as the k=1 loop: a judge that raises is a rejection."""

    def exploding_batch(slot, prefixes, lines):
        raise RuntimeError("judge server died")

    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(["- a.", "- b."]),
        judge_fn=accept_all,
        judge_batch_fn=exploding_batch,
        lookahead=2,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=10,
        max_line_chars=400,
    )
    assert result.lines == ["- fb."]
    assert result.stats["judge_errors"] == 2
    assert result.stats["fallback"] == 1


def test_lookahead_of_one_never_takes_the_batched_path():
    """The default must be the original loop, not a batch of size one."""
    batch = batch_of({"- a.": LineVerdict(True, 0.99, "judge_accept")})
    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(["- a."]),
        judge_fn=accept_all,
        judge_batch_fn=batch,
        lookahead=1,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=10,
        max_line_chars=400,
    )
    assert result.lines == ["- a."]
    assert batch.calls == [], "lookahead=1 must not reach judge_batch_fn"
    assert result.stats["judge_batches"] == 0


def test_lookahead_stops_the_batch_at_the_line_cap():
    """The cap bounds the speculation, not just the acceptance."""
    batch = batch_of({f"- {i}.": LineVerdict(True, 0.99, "judge_accept") for i in range(9)})
    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft([f"- {i}." for i in range(9)]),
        judge_fn=accept_all,
        judge_batch_fn=batch,
        lookahead=8,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=2,
        max_line_chars=400,
    )
    assert result.lines == ["- 0.", "- 1."]
    assert result.finished_reason == "line_cap"
    assert all(len(call) <= 2 for call in batch.calls), "never draft past the cap"


def test_lookahead_blank_line_still_costs_no_judge_call():
    """A blank line carries no claim, batched or not."""
    batch = batch_of(
        {
            "- a.": LineVerdict(True, 0.99, "judge_accept"),
            "- b.": LineVerdict(True, 0.99, "judge_accept"),
        }
    )
    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(["- a.", "\n", "- b."]),
        judge_fn=accept_all,
        judge_batch_fn=batch,
        lookahead=4,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=10,
        max_line_chars=400,
    )
    assert batch.calls == [["- a.", "- b."]], "the blank line is not judged"
    assert result.stats["judge_calls"] == 2


def test_lookahead_events_name_the_draft_call_that_produced_the_line():
    """The event's `call` is the join key to per-call token records.

    Under lookahead the events are emitted in the acceptance pass, after the
    whole batch has been drafted, so reading the live draft_calls counter there
    stamps every line in the batch with the LAST call's ordinal. The accepted
    text survives that, but every table built by joining tokens to decisions --
    per-line length by accept/reject, draft-vs-target provenance -- silently
    reads another line's record.
    """
    lines = ["- a.", "- b.", "- c."]
    batch = batch_of({line: LineVerdict(True, 0.99, "judge_accept") for line in lines})
    result = compact_slot(
        slot="goal",
        draft_fn=scripted_draft(lines),
        judge_fn=accept_all,
        judge_batch_fn=batch,
        lookahead=4,
        fallback_fn=lambda s, a: "- fb.",
        max_lines=10,
        max_line_chars=400,
    )
    drafted = [e for e in result.events if e["source"] == "draft"]
    assert [e["line"] for e in drafted] == lines
    assert [e["call"] for e in drafted] == [1, 2, 3], (
        "each line must carry its own draft ordinal, not the batch's last"
    )
    assert len({e["call"] for e in drafted}) == len(drafted), "ordinals must be distinct"
