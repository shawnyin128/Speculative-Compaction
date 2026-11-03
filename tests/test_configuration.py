"""Reject configuration mistakes before resource acquisition."""

import math
import subprocess
import sys
from pathlib import Path

import pytest

from hparam.hparam import Compaction, Endpoint, load_config
from training.training import llamafactory_arguments


def endpoints():
    return {
        "draft": Endpoint("http://localhost:8503/v1", "draft", "draft-tokenizer"),
        "target": Endpoint("http://localhost:8502/v1", "target", "target-tokenizer"),
    }


def test_zero_threshold_is_explicit_not_overridden_by_default():
    p = Compaction(**endpoints(), eta=0)
    p.validate()
    assert p.eta == 0
    p = Compaction(**endpoints(), verify_method="sr")
    p.validate()
    assert p.eta == 7


@pytest.mark.parametrize(
    "changes",
    [
        {"eta": math.nan},
        {"eta": 2},
        {"verify_method": "bad"},
        {"slot_concurrency": True},
        {"line_split": "bad"},
        {"verify_shared_prefix": True, "fallback_full_schema_prefix": False},
        {"max_slot_tokens": None},
        {"verify_lookahead": 2},
        {"timeout_seconds": 0},
    ],
)
def test_invalid_settings_fail_early(changes):
    with pytest.raises(ValueError):
        Compaction(**endpoints(), **changes).validate()


def test_unknown_yaml_keys_do_not_silently_change_defaults(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "draft: {url: http://localhost/v1, model: d, tokenizer: d}\nslot_concurreny: 6\n"
    )
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(path, "compact")


def test_all_shipped_presets_validate_and_training_keeps_sft_semantics():
    root = Path(__file__).resolve().parents[1]
    for name in ["gemma4", "qwen35"]:
        p = load_config(root / f"configs/compact/{name}.yaml", "compact")
        assert p.slot_concurrency == 6 and p.line_split == "client" and p.eta == 0.7
        assert p.target.tokenizer != p.draft.tokenizer
        recipe = llamafactory_arguments(
            load_config(root / f"configs/train/{name}.yaml", "train-sft")
        )
        assert recipe["lora_rank"] == 16 and recipe["learning_rate"] == 1e-4
        assert recipe["bf16"] and recipe["cutoff_len"] == 16384
        assert recipe["stage"] == "sft" and recipe["seed"] == 42


def test_installed_command_works_outside_source_tree_without_heavy_imports(tmp_path):
    command = [sys.executable, "-c", "import sys; from entrypoint import main; main(['--help'])"]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "prepare-sft" in completed.stdout
    code = "import sys; import entrypoint; assert not {'torch','transformers','datasets','llamafactory'}.intersection(sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], cwd=tmp_path).returncode == 0
