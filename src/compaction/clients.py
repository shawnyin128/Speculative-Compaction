"""HTTP clients binding the spec-compact engine to served models.

DraftLineClient: chunked decoding of the whole-slot policy -- prompt is the
TRAINING prompt (render_draft_prompt over build_shared_prefix +
build_slot_suffix, byte-identical; pinned by tests) plus the accepted lines,
generated one line at a time via stop=["\n"] on a completions endpoint.

No silent truncation anywhere: prompts over the token budget raise
PromptTooLongError (lesson: TRL max_prompt_length beheading).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from compaction.prompts import (
    SLOTS,
    build_shared_prefix,
    build_slot_suffix,
    render_draft_prompt,
)
from utils import measurement_clock as time


class PromptTooLongError(RuntimeError):
    """Prompt exceeds the configured token budget; refuse instead of truncating."""


@dataclass
class DraftLineClient:
    base_url: str
    model: str
    api_key: str
    context_text: str
    tokenizer: Any
    max_prompt_tokens: int
    max_line_tokens: int
    temperature: float
    top_p: Optional[float]
    timeout_seconds: float
    full_schema_prefix: bool
    capture_timing: bool
    line_split: str
    buffer_extra_lines: bool
    _line_buffer: dict = field(default_factory=dict, repr=False, compare=False)

    # Step 1: the per-slot base prompt. Single-slot schema is byte-identical to
    # training; full-schema shares the trajectory prefix across all six slots.
    def base_prompt(self, slot: str) -> str:
        slots = SLOTS if self.full_schema_prefix else (slot,)
        body = build_shared_prefix(self.context_text, slots=slots) + build_slot_suffix(slot)
        return render_draft_prompt(self.tokenizer, body)

    # Qwen tokenizes a single newline and a blank-line break as two DIFFERENT
    # tokens, and which one the model emits is the whole signal for "next line"
    # vs "new paragraph". Stopping on the text "\n" collapses them: both stop
    # the same way and the response cannot tell you which was meant.
    NEWLINE_TOKEN = 198  # "\n"   -- next line of the same block
    PARAGRAPH_TOKEN = 271  # "\n\n" -- blank line, i.e. a new block
    SEPARATOR_BY_STOP_TOKEN = {NEWLINE_TOKEN: "\n", PARAGRAPH_TOKEN: "\n\n"}

    @staticmethod
    def _newline_stop_token_map(tokenizer) -> dict[int, str]:
        """id -> "\\n" | "\\n\\n" for every vocab entry whose decoded text ends
        in a newline, for line_split="stop_tokens_expanded".

        Decoding all ~150k ids one at a time is a few seconds' work, so this
        is cached by tokenizer identity -- method.py loads one tokenizer per
        eval run and reuses it across every row, not one per row, and without
        the cache this would re-scan the vocab on every DraftLineClient built
        from it.
        """
        cached = DraftLineClient._NEWLINE_STOP_CACHE.get(id(tokenizer))
        if cached is not None:
            return cached
        mapping: dict[int, str] = {}
        for token_id in range(len(tokenizer)):
            text = tokenizer.decode([token_id], skip_special_tokens=False)
            if text.endswith("\n\n"):
                mapping[token_id] = "\n\n"
            elif text.endswith("\n"):
                mapping[token_id] = "\n"
        DraftLineClient._NEWLINE_STOP_CACHE[id(tokenizer)] = mapping
        return mapping

    # Keyed by id(tokenizer), not the tokenizer itself -- most tokenizer
    # objects are not hashable in a way that would dedupe correctly here, and
    # identity is what actually matters: two DraftLineClients sharing the same
    # loaded tokenizer object should share one scan.
    #
    # No type annotation, deliberately: an annotated assignment here would
    # make @dataclass treat this as a field, and a bare {} default on a
    # dataclass field raises at class-definition time (mutable default).
    # SEPARATOR_BY_STOP_TOKEN above avoids the same trap the same way.
    _NEWLINE_STOP_CACHE = {}

    # ---- line_split="client" -------------------------------------------
    # A buffered response is only usable while the slot's accepted_lines are
    # still exactly what the model saw when it wrote them. A rejected line
    # replaced by the target's repair invalidates every line generated after
    # it -- those were conditioned on text that is no longer in the summary --
    # so the snapshot is compared, not just the length.
    def _buffered(self, slot, accepted_lines):
        entry = self._line_buffer.get(slot)
        if not entry or not entry["pending"]:
            return None
        if entry["accepted"] != list(accepted_lines):
            self._line_buffer.pop(slot, None)
            return None
        line, done = entry["pending"].pop(0)
        entry["accepted"] = list(accepted_lines) + [line]
        if not entry["pending"]:
            self._line_buffer.pop(slot, None)
        return line, done

    @staticmethod
    def _split_lines(text: str, finished_on_eos: bool) -> list[tuple[str, bool]]:
        """Text -> [(line, done)], under this module's separator contract.

        A blank-line break is carried as a trailing newline on the line before
        it, so rejoining the result with a newline reproduces the original byte
        for byte -- the same invariant the stop-token path maintains, and what
        next_line's prompt rebuild depends on.

        Without eos the text stops wherever max_tokens fell, so the segment
        after the last newline is NOT a finished line and is dropped. Keeping
        it is exactly what puts mid-sentence fragments into the summary.
        """
        newline = chr(10)
        if not text:
            return []
        segments = text.split(newline)
        if not finished_on_eos:
            segments.pop()  # incomplete tail, never a line
        lines: list[str] = []
        index = 0
        while index < len(segments):
            body = segments[index]
            blank_follows = index + 1 < len(segments) and not segments[index + 1].strip()
            if blank_follows and body.strip():
                lines.append(body + newline)  # carry the blank line
                index += 2
            else:
                lines.append(body)
                index += 1
        lines = [line for line in lines if line.strip()]
        if not lines:
            return []
        return [(line, False) for line in lines[:-1]] + [(lines[-1], bool(finished_on_eos))]

    # Step 2: one chunked-decode step -- returns (line | None, done).
    def next_line(
        self, slot: str, accepted_lines: list[str], usage_out: dict | None = None
    ) -> tuple[Optional[str], bool]:
        # Rebuild EXACTLY what the model has generated so far. accepted_lines
        # carries its own separators (a paragraph break is stored as a trailing
        # "\n" on the line before it), so joining with "\n" reproduces the
        # original byte-for-byte and nothing is appended on the end.
        #
        # The old code appended one "\n" after every accepted line. That is the
        # inverse of stop=["\n"] -- which swallows the separator -- so it looks
        # right, and for an ordinary line-to-line boundary it is. It is wrong
        # exactly where the model wanted something else: a blank line before a
        # new "### " block, or nothing at all before <|im_end|>. Measured on
        # sy128/SpecComp-slot-sft-qwen3-14B with a real progress-slot row:
        # ending the prompt with the extra "\n" made the draft invent an extra
        # "- [x] Confirmed the new reservation details and payment method"
        # bullet instead of breaking the paragraph, and invent "- [None]"
        # instead of stopping. Ending it without the extra "\n" produced the
        # paragraph break and the eos, both correct. So the appended newline
        # was not a formatting detail -- it was making the draft hallucinate a
        # list item at every block boundary and at the end of every slot, which
        # is also why slots ran to the line cap instead of terminating.
        if self.line_split == "client":
            hit = self._buffered(slot, accepted_lines)
            if hit is not None:
                # Served from the previous response: no request, so no usage
                # to report. by_role therefore shows fewer draft calls each
                # carrying more tokens, which is the point of the mode.
                return hit

        prompt = self.base_prompt(slot) + "\n".join(accepted_lines)

        n_tokens = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tokens > self.max_prompt_tokens:
            raise PromptTooLongError(
                f"[ERROR] draft prompt {n_tokens} tokens > budget {self.max_prompt_tokens}"
            )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_line_tokens,
            "temperature": self.temperature,
            # Stop on the TOKENS, not on the text. vLLM reports the id of
            # the stop token in stop_reason, so the response says which
            # separator the model chose; a text stop would report "\n" for
            # both and the distinction would be gone.
            "stop_token_ids": [self.NEWLINE_TOKEN, self.PARAGRAPH_TOKEN],
        }
        separator_map = self.SEPARATOR_BY_STOP_TOKEN
        if self.line_split == "client":
            # No server-side stop at all: the boundary is found in the decoded
            # text below, where a merged newline is as visible as a bare one.
            payload.pop("stop_token_ids")
        elif self.line_split == "stop_tokens_expanded":
            # Same idea, but stop on every merged newline id too, not just the
            # bare 198/271 -- see the mode's docstring above for what this
            # buys and why it is not also a win for the draft.
            separator_map = self._newline_stop_token_map(self.tokenizer)
            payload["stop_token_ids"] = list(separator_map)
        elif self.line_split != "stop_tokens":
            raise ValueError(
                f"[ERROR] line_split={self.line_split!r}; expected "
                "'stop_tokens', 'stop_tokens_expanded', or 'client'."
            )
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.capture_timing:
            body, ttft_seconds, decode_seconds = self._post_streaming(payload)
        else:
            response = requests.post(
                f"{self.base_url}/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            ttft_seconds = decode_seconds = None
        # Chunked decoding re-sends the whole prompt every line, so per-call
        # prompt_tokens is the only place that cost shows up; a run that sums
        # only generated tokens understates the line path badly.
        #
        # cached_prompt_tokens is what makes that number readable. The resend is
        # the SAME prefix every time, so with APC on nearly all of it is a cache
        # hit and costs no prefill -- prompt_tokens alone reads as a catastrophe
        # that the server never actually paid for. Serving under YaRN turns APC
        # off entirely (scripts/serve_long_prefill.sh), which is exactly the
        # regime where the two numbers diverge and the distinction stops being
        # academic. vLLM reports it; nothing here read it before.
        if usage_out is not None:
            usage = body.get("usage") or {}
            usage_out["prompt_tokens"] = usage.get("prompt_tokens")
            usage_out["completion_tokens"] = usage.get("completion_tokens")
            usage_out["cached_prompt_tokens"] = (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            )
            usage_out["ttft_seconds"] = ttft_seconds
            usage_out["decode_seconds"] = decode_seconds
        choice = body["choices"][0]
        if self.line_split == "client":
            raw = choice.get("text") or ""
            # eos, not a stop token: nothing was asked to stop this
            # generation, so finish_reason "stop" can only be the slot ending.
            finished_on_eos = choice.get("finish_reason") == "stop"
            produced = self._split_lines(raw, finished_on_eos)
            if not produced:
                # Not one newline in the whole budget. Returning nothing would
                # spin the slot loop forever, so fall back to what the
                # stop-token path does on a max_tokens hit.
                stripped = raw.rstrip(chr(10))
                return (stripped if stripped.strip() else None), finished_on_eos
            line, done = produced[0]
            rest = produced[1:]
            if rest and self.buffer_extra_lines:
                self._line_buffer[slot] = {
                    "accepted": list(accepted_lines) + [line],
                    "pending": rest,
                }
            return line, done
        # rstrip("\n") only: leading whitespace is load-bearing for nested list
        # items, which the slot contracts explicitly allow.
        text = choice["text"].rstrip("\n")
        separator = separator_map.get(choice.get("stop_reason"))

        # Whitespace-only is empty for the purpose of "did the model say
        # anything", even though the text itself keeps its leading whitespace
        # when it does.
        has_content = bool(text.strip())

        # A null stop_reason with finish_reason "stop" means the model emitted
        # eos: the slot is complete, keep this final line and end.
        if choice.get("finish_reason") == "stop" and choice.get("stop_reason") is None:
            return (text if has_content else None), True
        if separator is None:  # hit max_line_tokens, or an unknown stop
            return (text if has_content else None), False
        # Carry the paragraph break with the line it follows, so that rejoining
        # with "\n" reproduces "\n\n" and the blank line survives into the
        # assembled slot. An empty text here is a legitimate blank line, not a
        # finished slot -- the old code read the two as the same thing.
        return text + separator[1:], False

    def _post_streaming(self, payload: dict) -> tuple[dict, Optional[float], Optional[float]]:
        """Stream ``/completions`` to time prefill (TTFT) vs decode.

        Reassembles the SSE stream into the same ``{"choices": [...],
        "usage": {...}}`` shape ``next_line`` already parses non-streaming,
        so the caller's post-processing does not need two code paths.
        prefill_seconds is wall time to the first content token (queueing +
        prefill are indistinguishable from the client side, same as any
        TTFT measurement); decode_seconds is everything after that.
        """
        stream_payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        started = time.perf_counter()
        first_token_at: Optional[float] = None
        text_parts: list[str] = []
        finish_reason = stop_reason = None
        usage: dict = {}
        response = requests.post(
            f"{self.base_url}/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=stream_payload,
            timeout=self.timeout_seconds,
            stream=True,
        )
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data = raw_line[len("data:") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if choices:
                piece = choices[0].get("text") or ""
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(piece)
                if choices[0].get("finish_reason") is not None:
                    finish_reason = choices[0]["finish_reason"]
                if choices[0].get("stop_reason") is not None:
                    stop_reason = choices[0]["stop_reason"]
            if chunk.get("usage"):
                usage = chunk["usage"]
        ended = time.perf_counter()
        ttft_seconds = (first_token_at - started) if first_token_at is not None else None
        decode_seconds = (ended - first_token_at) if first_token_at is not None else None
        body = {
            "choices": [
                {
                    "text": "".join(text_parts),
                    "finish_reason": finish_reason,
                    "stop_reason": stop_reason,
                }
            ],
            "usage": usage,
        }
        return body, ttft_seconds, decode_seconds


@dataclass
class TargetFallbackClient(DraftLineClient):
    """Generate one trusted repair line using the target's own tokenizer.

    Target repairs stay greedy and never reuse unverified extra output lines.
    """

    buffer_extra_lines: bool = False

    def fallback_line(
        self, slot: str, accepted_lines: list[str], usage_out: dict | None = None
    ) -> str:
        line, done = self.next_line(slot, accepted_lines, usage_out=usage_out)
        # `done` means the target emitted eos, and next_line's own contract for
        # that case is "keep this final line and end" -- it already returns None
        # instead of the text when there was no content. Discarding the line
        # because `done` is set threw away a repair the target had just written
        # and then failed the empty string against the hard gate, which ends the
        # slot as "fallback_rejected". Measured on job 16592850 that was 12.5% of
        # 8k slot runs and 23.6% of 16k ones -- every one of them losing a
        # written line AND every line the slot would have gone on to write. The
        # draft path never had this bug (see method.py's "done=True with a line
        # means the draft emitted eos right after this final line: gate it like
        # any other, then end the slot").
        #
        # The slot is NOT ended here: fallback_fn's contract is str, with no
        # channel for "and we are finished". Keeping the line lets the loop ask
        # the draft once more, and the draft -- looking at a slot the target
        # just completed -- ends it. One extra draft call, no lost text.
        #
        # A bare line break is not a finished slot. The rebuilt prompt ends
        # without the previous line's newline, so the target's first token is
        # usually that newline and the stop fires on it with no content; the
        # empty string then failed the hard gate and ended the slot. Record the
        # break as the draft path does (an entry holding just the separator)
        # and ask for the line itself, carrying a paragraph break in front of it.
        if line is not None and not line.strip() and not done:
            retry_usage: dict | None = {} if usage_out is not None else None
            content, _ = self.next_line(slot, list(accepted_lines) + [line], usage_out=retry_usage)
            if usage_out is not None:
                for key, value in retry_usage.items():
                    if isinstance(value, (int, float)) and isinstance(
                        usage_out.get(key), (int, float)
                    ):
                        usage_out[key] += value
                    elif usage_out.get(key) is None:
                        usage_out[key] = value
            return line + content if content is not None and content.strip() else ""
        return "" if line is None else line
