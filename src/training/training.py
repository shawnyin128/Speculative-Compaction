"""Compose the project SFT recipe with the isolated LLaMA-Factory module."""

import json
from dataclasses import asdict
from pathlib import Path

from hparam.hparam import Training


def llamafactory_arguments(parameters: Training) -> dict:
    """Translate the one owned training schema to LLaMA-Factory's SFT interface."""
    parameters.validate()
    return {
        **asdict(parameters),
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "dataset": "slot_sft_pretokenized",
        "val_size": 0.01,
        "template": "qwen3",
        "save_strategy": "epoch",
        "plot_loss": True,
        "lr_scheduler_type": "cosine",
        "do_eval": True,
        "eval_strategy": "steps",
    }


def train_sft(parameters: Training):
    """Run the real trainer against pretokenized data, preserving existing outputs.

    The tokenized dataset already owns its trace-disjoint validation partition;
    LLaMA-Factory's val_size above is a parser placeholder, not a second split.
    """
    arguments = llamafactory_arguments(parameters)
    dataset_path = Path(parameters.tokenized_path)
    manifest = dataset_path / "dataset_dict.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            "Build or download the tokenized DatasetDict first: " + str(manifest)
        )
    splits = json.loads(manifest.read_text(encoding="utf8"))["splits"]
    if not {"train", "validation"}.issubset(splits):
        raise ValueError("SFT requires the prebuilt train and validation partitions.")
    output = Path(parameters.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new training output directory: " + str(output))
    # The optional trainer imports torch and checks its own installed versions.
    from llamafactory.train.tuner import run_exp

    output.mkdir(parents=True, exist_ok=True)
    (output / "speccompact_config.json").write_text(
        json.dumps(arguments, indent=2) + "\n", encoding="utf8"
    )
    run_exp(args=arguments)
