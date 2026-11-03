"""Check supervised-target integrity and the real trainer's input boundary."""

import json
import sys
from types import ModuleType

import pytest

from data.encoding import encode_slot
from hparam.hparam import Data, Training
from training.training import train_sft


class Tokenizer:
    def __call__(self, text, **kwargs):
        pieces = text.split("<|im_end|>")
        ids = []
        for index, piece in enumerate(pieces):
            if index:
                ids.append(2)
            ids.extend(ord(char) + 1000 for char in piece)
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join("<|im_end|>" if i == 2 else chr(i - 1000) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "</user><assistant>"


def row(context="short context", target="- Keep the confirmed constraint."):
    return {
        "slot": "goal",
        "context_text": context,
        "slot_target": target,
        "from": "example",
    }


def test_supervision_masks_only_the_prompt_and_keeps_complete_target_and_eos():
    tokenizer = Tokenizer()
    sample, diagnostics = encode_slot(row(), Data(cutoff_len=10000), tokenizer)
    prompt = diagnostics["prompt_tokens"]
    expected = tokenizer(row()["slot_target"] + "<|im_end|>\n")["input_ids"]
    assert sample["labels"][:prompt] == [-100] * prompt
    assert sample["labels"][prompt:] == expected
    assert sample["input_ids"][prompt:] == expected
    assert len(sample["input_ids"]) == len(sample["labels"])


def test_overlong_target_is_dropped_whole():
    sample, diagnostics = encode_slot(row(target="x" * 21), Data(max_target_tokens=20), Tokenizer())
    assert sample is None
    assert diagnostics["counts"] == {"dropped_long_target": 1}


def test_context_can_shrink_without_shortening_supervised_target():
    tokenizer = Tokenizer()
    original, _ = encode_slot(row("x" * 2000), Data(cutoff_len=10000), tokenizer)
    limit = len(original["input_ids"]) - 500
    sample, diagnostics = encode_slot(row("x" * 2000), Data(cutoff_len=limit), tokenizer)
    assert len(sample["input_ids"]) <= limit
    assert diagnostics["counts"]["truncated"] == 1
    assert (
        sample["labels"][diagnostics["prompt_tokens"] :]
        == tokenizer(row()["slot_target"] + "<|im_end|>\n")["input_ids"]
    )


def test_context_floor_rejects_instead_of_damaging_target():
    sample, diagnostics = encode_slot(row(), Data(cutoff_len=50), Tokenizer())
    assert sample is None
    assert diagnostics["counts"]["dropped_context_floor"] == 1


def test_trainer_rejects_a_dataset_without_held_out_partition(tmp_path):
    (tmp_path / "dataset_dict.json").write_text(json.dumps({"splits": ["train"]}))
    with pytest.raises(ValueError, match="train and validation"):
        train_sft(Training(tokenized_path=str(tmp_path)))


def test_training_calls_framework_with_pretokenized_recipe_and_records_settings(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset_dict.json").write_text(json.dumps({"splits": ["train", "validation"]}))
    output = tmp_path / "output"
    calls = []
    tuner = ModuleType("llamafactory.train.tuner")
    tuner.run_exp = lambda *, args: calls.append(args)
    monkeypatch.setitem(sys.modules, "llamafactory.train.tuner", tuner)
    parameters = Training(tokenized_path=str(dataset), output_dir=str(output))
    train_sft(parameters)
    assert len(calls) == 1
    recipe = calls[0]
    assert recipe["tokenized_path"] == str(dataset)
    assert recipe["do_train"] and recipe["do_eval"] and recipe["stage"] == "sft"
    assert recipe == json.loads((output / "speccompact_config.json").read_text())
    with pytest.raises(FileExistsError):
        train_sft(parameters)
    assert len(calls) == 1
