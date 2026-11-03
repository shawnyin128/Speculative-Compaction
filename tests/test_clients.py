"""Tests for the spec-compact HTTP clients (chunked draft decode + fallback)."""

import pytest

from compaction.clients import DraftLineClient, PromptTooLongError, TargetFallbackClient
from compaction.prompts import (
    SLOTS,
    build_shared_prefix,
    build_slot_suffix,
)


class FakeTokenizer:
    """No chat template: render_draft_prompt falls back to the raw prompt."""

    chat_template = None

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}


def make_client(cls=DraftLineClient, **overrides):
    defaults = dict(
        base_url="http://fake:1/v1",
        model="draft",
        api_key="EMPTY",
        context_text="[user] book a flight",
        tokenizer=FakeTokenizer(),
        max_prompt_tokens=10_000,
        temperature=0.0,
        top_p=None,
        timeout_seconds=120,
        full_schema_prefix=False,
        capture_timing=False,
        line_split="stop_tokens",
        buffer_extra_lines=cls is DraftLineClient,
        max_line_tokens=64,
    )
    defaults.update(overrides)
    return cls(**defaults)


@pytest.mark.parametrize("reuse,expected_requests", [(True, 1), (False, 2)])
def test_client_buffer_ablation_changes_round_trips_without_losing_lines(
    monkeypatch, reuse, expected_requests
):
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        text = "- first\n- second\n" if len(calls) == 1 else "- second\n"

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [{"text": text, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 6},
                }

        return Response()

    monkeypatch.setattr("requests.post", post)
    client = make_client(line_split="client", buffer_extra_lines=reuse)
    assert client.next_line("goal", []) == ("- first", False)
    assert client.next_line("goal", ["- first"]) == ("- second\n", True)
    assert len(calls) == expected_requests


def test_base_prompt_pins_to_training_builders():
    client = make_client()
    expected = build_shared_prefix("[user] book a flight", slots=("goal",)) + build_slot_suffix(
        "goal"
    )
    assert client.base_prompt("goal") == expected  # byte-equal to training


def test_full_schema_prefix_shares_trajectory_across_slots():
    """With full_schema_prefix, the prefix up to the trajectory is identical for
    every slot (only the post-trajectory suffix diverges) -- the property APC
    needs to cache the ~8k context once and reuse it for all six slots."""
    client = make_client(full_schema_prefix=True)
    prompts = {slot: client.base_prompt(slot) for slot in SLOTS}
    shared = build_shared_prefix("[user] book a flight", slots=SLOTS)
    for slot, prompt in prompts.items():
        # full schema prefix (incl. trajectory) is a common prefix of every slot
        assert prompt.startswith(shared)
        # and the divergence sits AFTER the shared prefix, in the slot suffix
        assert build_slot_suffix(slot) in prompt[len(shared) :]
    # distinct suffixes -> distinct final prompts, same cacheable head
    assert len(set(prompts.values())) == len(SLOTS)


def test_full_schema_prefix_differs_from_training_default():
    """The knob is opt-in: default stays single-slot (training-pinned)."""
    default = make_client().base_prompt("goal")
    full = make_client(full_schema_prefix=True).base_prompt("goal")
    assert default != full


def test_fallback_client_takes_the_flag_independently_of_the_draft():
    """The target can share one prefix across slots even when the draft cannot.

    Nothing about the target is pinned to the SFT format, so this is the side
    the flag is safe on -- and the side where it pays, because the target is
    also serving every judge call and only sees the compaction prompt when a
    line is rejected. Off, the first fallback of each slot re-prefills the whole
    context (0.87s at 8k, 1.89s at 16k, job 16592850); on, one prefill per
    context serves all six.
    """
    off = make_client(cls=TargetFallbackClient)
    on = make_client(cls=TargetFallbackClient, full_schema_prefix=True)
    shared = build_shared_prefix("[user] book a flight", slots=SLOTS)

    assert not all(off.base_prompt(slot).startswith(shared) for slot in SLOTS)
    assert all(on.base_prompt(slot).startswith(shared) for slot in SLOTS)


def test_prompt_grows_monotonically_with_accepted_lines(monkeypatch):
    client = make_client()
    seen = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append(json["prompt"])

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"text": "- next line", "finish_reason": "stop"}]}

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    client.next_line("goal", [])
    client.next_line("goal", ["- a."])
    client.next_line("goal", ["- a.", "- b."])
    assert seen[0] == client.base_prompt("goal")
    assert seen[1].startswith(seen[0]) and seen[1].endswith("- a.")
    assert seen[2].startswith(seen[1]) and seen[2].endswith("- a.\n- b.")
    # No trailing separator is appended: the draft chooses its own, and the
    # choice (newline vs blank line vs eos) is what stop_token_ids reports
    # back. Appending one here is what made the draft invent an extra list
    # item at every block boundary.
    assert not seen[1].endswith("\n")
    assert not seen[2].endswith("\n")


def test_empty_completion_means_slot_done(monkeypatch):
    client = make_client()

    def fake_post(url, **kwargs):
        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"text": "  ", "finish_reason": "stop"}]}

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    line, done = client.next_line("goal", [])
    assert done and line is None


def test_prompt_over_limit_raises_not_truncates():
    client = make_client(max_prompt_tokens=3)
    with pytest.raises(PromptTooLongError):
        client.next_line("goal", [])


def _client_with_response(monkeypatch, payload, cls=None):
    client = make_client(cls=cls) if cls is not None else make_client()

    def fake_post(url, **kwargs):
        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return payload

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    return client


def test_eos_after_final_line_ends_slot(monkeypatch):
    client = _client_with_response(
        monkeypatch,
        {"choices": [{"text": "- last point", "finish_reason": "stop", "stop_reason": None}]},
    )
    line, done = client.next_line("goal", [])
    assert line == "- last point" and done is True


def test_stop_string_hit_means_more_lines_follow(monkeypatch):
    client = _client_with_response(
        monkeypatch,
        {
            "choices": [
                {
                    "text": "- mid point",
                    "finish_reason": "stop",
                    "stop_reason": DraftLineClient.NEWLINE_TOKEN,
                }
            ]
        },
    )
    line, done = client.next_line("goal", [])
    assert line == "- mid point" and done is False


def test_paragraph_token_is_carried_back_as_a_blank_line(monkeypatch):
    """A "\n\n" stop must survive as a blank line, not collapse into "\n".

    Qwen emits the blank-line break as its own token, and that token is the
    only signal separating "next bullet" from "new ### block". Stopping on the
    text "\n" reports the same thing for both, so the block structure of any
    slot whose format has sections (progress) is lost -- silently, because the
    draft still returns plausible lines either way.
    """
    client = _client_with_response(
        monkeypatch,
        {
            "choices": [
                {
                    "text": "- [x] done",
                    "finish_reason": "stop",
                    "stop_reason": DraftLineClient.PARAGRAPH_TOKEN,
                }
            ]
        },
    )
    line, done = client.next_line("progress", [])
    assert done is False
    assert line == "- [x] done\n"
    # Rejoining is what turns the marker back into the blank line.
    assert "\n".join([line, "### In Progress"]) == "- [x] done\n\n### In Progress"


def test_stops_on_tokens_not_on_text(monkeypatch):
    """The request must use stop_token_ids; a text stop cannot tell them apart."""
    client = make_client()
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(json)

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "text": "x",
                            "finish_reason": "stop",
                            "stop_reason": DraftLineClient.NEWLINE_TOKEN,
                        }
                    ]
                }

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    client.next_line("goal", [])
    assert sent["stop_token_ids"] == [
        DraftLineClient.NEWLINE_TOKEN,
        DraftLineClient.PARAGRAPH_TOKEN,
    ]
    assert "stop" not in sent


def test_leading_indentation_survives(monkeypatch):
    """Nested list items are allowed by the slot contracts, so leading
    whitespace is content -- only the trailing separator may be stripped."""
    client = _client_with_response(
        monkeypatch,
        {
            "choices": [
                {
                    "text": "    - nested item",
                    "finish_reason": "stop",
                    "stop_reason": DraftLineClient.NEWLINE_TOKEN,
                }
            ]
        },
    )
    line, _ = client.next_line("goal", [])
    assert line == "    - nested item"


def test_cached_prompt_tokens_are_read_from_usage(monkeypatch):
    """The growing-prompt resend is only affordable because it is cached.

    next_line re-sends base_prompt + every accepted line on EVERY line, so
    prompt_tokens grows linearly and reads as ruinous prefill cost. What the
    server actually recomputed is prompt_tokens - cached_tokens; vLLM reports
    the second number and nothing here read it until now. It goes to zero the
    moment prefix caching is off (any YaRN-served run), which is exactly when
    the distinction stops being academic.
    """

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [{"text": "- next line", "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 8,
                        "prompt_tokens_details": {"cached_tokens": 960},
                    },
                }

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    usage = {}
    make_client().next_line("goal", ["- a."], usage_out=usage)
    assert usage["prompt_tokens"] == 1000
    assert usage["cached_prompt_tokens"] == 960


def test_cached_prompt_tokens_missing_is_none_not_zero(monkeypatch):
    """A backend that omits the details block has not reported a 0% hit rate."""

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [{"text": "- next line", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 8},
                }

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    usage = {}
    make_client().next_line("goal", [], usage_out=usage)
    assert usage["cached_prompt_tokens"] is None


def test_fallback_keeps_the_final_line_when_the_target_stops(monkeypatch):
    """A repair line followed by eos must survive.

    next_line's contract for eos is "keep this final line and end", and it
    already returns None when there was no content. fallback_line used to
    discard the text whenever `done` was set, so the target's repair was lost
    and the empty string then failed the hard gate as `empty_line` -- ending
    the slot as `fallback_rejected`. On job 16592850 that fired on 12.5% of 8k
    slot runs and 23.6% of 16k ones.
    """
    client = _client_with_response(
        monkeypatch,
        {"choices": [{"text": "- target repair.", "finish_reason": "stop", "stop_reason": None}]},
        cls=TargetFallbackClient,
    )
    assert client.fallback_line("goal", []) == "- target repair."


def test_fallback_returns_empty_only_when_there_is_no_content(monkeypatch):
    """Whitespace-only really is nothing to repair with, eos or not."""
    client = _client_with_response(
        monkeypatch,
        {"choices": [{"text": "   ", "finish_reason": "stop", "stop_reason": None}]},
        cls=TargetFallbackClient,
    )
    assert client.fallback_line("goal", []) == ""


def _client_with_responses(monkeypatch, payloads, cls=TargetFallbackClient):
    """Serve ``payloads`` in order and record every request body sent."""
    client = make_client(cls=cls)
    sent = []

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        payload = payloads[len(sent) - 1]

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return payload

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    return client, sent


def _stop(text, token):
    return {
        "choices": [{"text": text, "finish_reason": "stop", "stop_reason": token}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 1 + len(text.split())},
    }


def test_fallback_asks_again_after_a_bare_line_break(monkeypatch):
    """The rebuilt prompt ends without the previous line's newline, so the
    target's first token is that newline and the stop fires on it with no
    content. That is not a finished slot: the line itself comes next."""
    client, sent = _client_with_responses(
        monkeypatch,
        [
            _stop("", DraftLineClient.NEWLINE_TOKEN),
            _stop("- repaired line", DraftLineClient.NEWLINE_TOKEN),
        ],
    )
    assert client.fallback_line("goal", ["- first"]) == "- repaired line"
    assert len(sent) == 2
    assert sent[0]["prompt"].endswith("- first")
    assert sent[1]["prompt"].endswith("- first\n")


def test_fallback_keeps_a_paragraph_break_before_the_repair(monkeypatch):
    client, sent = _client_with_responses(
        monkeypatch,
        [
            _stop("", DraftLineClient.PARAGRAPH_TOKEN),
            _stop("### In Progress", DraftLineClient.NEWLINE_TOKEN),
        ],
    )
    line = client.fallback_line("progress", ["- [x] done"])
    assert sent[1]["prompt"].endswith("- [x] done\n\n")
    # Rejoining with "\n" reproduces the blank line the target asked for.
    assert "\n".join(["- [x] done", line]) == "- [x] done\n\n### In Progress"


def test_fallback_gives_up_when_the_retry_is_blank_too(monkeypatch):
    client, sent = _client_with_responses(
        monkeypatch,
        [
            _stop("", DraftLineClient.NEWLINE_TOKEN),
            _stop("", DraftLineClient.NEWLINE_TOKEN),
        ],
    )
    assert client.fallback_line("goal", ["- first"]) == ""
    assert len(sent) == 2


def test_fallback_retry_adds_to_the_usage_record(monkeypatch):
    client, _ = _client_with_responses(
        monkeypatch,
        [
            _stop("", DraftLineClient.NEWLINE_TOKEN),
            _stop("- repaired line", DraftLineClient.NEWLINE_TOKEN),
        ],
    )
    usage = {}
    client.fallback_line("goal", ["- first"], usage_out=usage)
    assert usage["prompt_tokens"] == 200
    assert usage["completion_tokens"] == 5
