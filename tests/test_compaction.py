"""Complete draft/accept/reject/repair flow with deterministic model responses."""

import json
import re

from compaction.compaction import Compactor
from compaction.prompts import SLOTS, build_shared_prefix, build_slot_suffix, render_draft_prompt
from hparam.hparam import Compaction, Endpoint


class Tokenizer:
    chat_template = "test chat template"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, messages, **kwargs):
        return (
            "".join("<" + m["role"] + ">" + m["content"] + "</" + m["role"] + ">" for m in messages)
            + "<assistant>"
        )


def model_response(url, headers, json, **kwargs):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            usage = {
                "prompt_tokens": 100,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 80},
            }
            if url.endswith("/chat/completions"):
                content = json["messages"][-1]["content"]
                label = (
                    "negative" if "Proposed next line:\n- unsupported" in content else "positive"
                )
                return {
                    "choices": [
                        {
                            "logprobs": {
                                "content": [{"top_logprobs": [{"token": label, "logprob": -0.01}]}]
                            }
                        }
                    ],
                    "usage": usage,
                }
            slot = re.search(r"Generate only the `([^`]+)`", json["prompt"]).group(1)
            if json["model"] == "target":
                text = "- repaired " + slot
            elif "- repaired " + slot in json["prompt"]:
                text = ""
            elif json["model"] == "draft":
                text = "- kept " + slot + "\n- unsupported " + slot + "\n"
            else:
                raise AssertionError("Unexpected model")
            return {"choices": [{"text": text, "finish_reason": "stop"}], "usage": usage}

    return Response()


def test_end_to_end_repair_changes_the_next_draft_prefix_and_keeps_accounting(monkeypatch):
    monkeypatch.setattr("requests.post", model_response)
    parameters = Compaction(
        draft=Endpoint("http://fake/v1", "draft", "d"),
        target=Endpoint("http://fake/v1", "target", "t"),
        fallback_line_split="client",
        max_prompt_tokens=100000,
        max_slot_tokens=None,
        summary_token_budget=None,
        capture_timing=False,
        record_llm_calls=True,
    )
    tokenizer = Tokenizer()
    result = Compactor(parameters, draft_tokenizer=tokenizer, target_tokenizer=tokenizer).compact(
        "original full context"
    )
    assert result.metadata["accepted_lines"] == 6
    assert result.metadata["fallback_lines"] == 6
    assert result.metadata["accept_rate"] == 0.5
    assert "- unsupported" not in result.text
    for slot in SLOTS:
        assert "- kept " + slot in result.text
        assert "- repaired " + slot in result.text
        events = result.metadata["slot_events"][slot]
        assert any(e["decision"] == "rejected" for e in events)
    total = result.metadata["compaction_llm_tokens"]["completion"]
    assert total == sum(
        v["completion_tokens"] for v in result.metadata["compaction_llm_tokens_by_role"].values()
    )
    json.dumps(result.metadata, allow_nan=False)


def test_training_prompt_is_the_exact_inference_prompt_and_uses_one_slot_schema():
    tokenizer = Tokenizer()
    for slot in SLOTS:
        user = build_shared_prefix("CONTEXT", slots=(slot,)) + build_slot_suffix(slot)
        expected = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert render_draft_prompt(tokenizer, user) == expected
        schema = user.split("contains these sections:\n")[1].split("\n\nConversation:")[0]
        assert len(schema.splitlines()) == 1
        assert user.endswith("Nested headings required by the slot contract are allowed.\n")


def test_draft_and_repair_prefix_controls_are_independent():
    p = Compaction(
        draft=Endpoint("http://fake/v1", "draft", "d"),
        target=Endpoint("http://fake/v1", "target", "t"),
        full_schema_prefix=False,
        fallback_full_schema_prefix=True,
    )
    p.validate()
    assert not p.full_schema_prefix and p.fallback_full_schema_prefix
