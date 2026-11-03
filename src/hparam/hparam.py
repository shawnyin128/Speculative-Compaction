"""The single typed configuration contract; validation acquires no resources."""

import argparse
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal

import yaml


def application_parser():
    """Own CLI names, help and types; the application parses this parser once."""
    parser = argparse.ArgumentParser(
        description="SpecCompact: draft, verify and repair agent context."
    )
    parser.add_argument("operation", choices=["compact", "prepare-sft", "train-sft"])
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Typed YAML configuration for the selected operation.",
    )
    parser.add_argument("--input", type=Path, help="UTF-8 context text; required for compact.")
    parser.add_argument("--output", type=Path, help="New JSON result path; required for compact.")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print validated effective settings without loading models or data.",
    )
    return parser


@dataclass
class Endpoint:
    url: str = field(metadata={"help": "OpenAI-compatible base URL ending in /v1."})
    model: str = field(metadata={"help": "Served model alias."})
    tokenizer: str = field(metadata={"help": "Matching tokenizer directory or model ID."})
    api_key_env: str = field(
        default="VLLM_API_KEY",
        metadata={"help": "Credential environment variable; never a secret value."},
    )

    def validate(self):
        if not self.url.startswith(("http://", "https://")) or not self.url.rstrip("/").endswith(
            "/v1"
        ):
            raise ValueError("Endpoint url must be an http(s) base URL ending in /v1.")
        if not self.model or not self.tokenizer or not self.api_key_env:
            raise ValueError("Endpoint model, tokenizer and API-key environment name are required.")


@dataclass
class Compaction:
    draft: Endpoint
    target: Endpoint | None = None
    verify: bool = True
    verify_method: Literal["dreamr", "sr"] = "dreamr"
    eta: float | None = None
    slot_concurrency: int = 6
    max_lines: int = 12
    max_line_chars: int = 400
    max_prompt_tokens: int = 28000
    max_line_tokens: int = 64
    max_slot_tokens: int | None = 320
    summary_token_budget: int | None = 2048
    temperature: float = 0.0
    top_p: float | None = None
    full_schema_prefix: bool = False
    fallback_full_schema_prefix: bool = True
    verify_shared_prefix: bool = True
    structural_bypass: bool = True
    line_split: Literal["client", "stop_tokens", "stop_tokens_expanded"] = "client"
    fallback_line_split: Literal["client", "stop_tokens", "stop_tokens_expanded"] = (
        "stop_tokens_expanded"
    )
    buffer_extra_lines: bool = True
    verify_schedule: Literal["sequential", "parallel"] = "sequential"
    verify_lookahead: int = 1
    timeout_seconds: float = 120
    capture_timing: bool = True
    record_llm_calls: bool = False

    def validate(self):
        self.draft.validate()
        if self.verify and self.target is None:
            raise ValueError("Verified compaction requires a target endpoint.")
        if self.target is not None:
            self.target.validate()
        if self.verify_method == "dreamr":
            threshold = 0.7 if self.eta is None else self.eta
            limit = 1
        elif self.verify_method == "sr":
            threshold = 7.0 if self.eta is None else self.eta
            limit = 9
        else:
            raise ValueError("verify_method must be dreamr or sr.")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 <= threshold <= limit
        ):
            raise ValueError("eta is outside the selected verifier scale.")
        self.eta = float(threshold)
        for name in [
            "slot_concurrency",
            "max_lines",
            "max_line_chars",
            "max_prompt_tokens",
            "max_line_tokens",
            "verify_lookahead",
        ]:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer.")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive.")
        if (
            type(self.temperature) not in (int, float)
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature
        ):
            raise ValueError("temperature must be nonnegative.")
        if self.top_p is not None and (
            type(self.top_p) not in (int, float) or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be in (0,1].")
        for name in [
            "verify",
            "full_schema_prefix",
            "fallback_full_schema_prefix",
            "verify_shared_prefix",
            "structural_bypass",
            "buffer_extra_lines",
            "capture_timing",
            "record_llm_calls",
        ]:
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + " must be boolean.")
        if self.verify_shared_prefix and not self.fallback_full_schema_prefix:
            raise ValueError("verify_shared_prefix requires fallback_full_schema_prefix.")
        for value in [self.line_split, self.fallback_line_split]:
            if value not in ("client", "stop_tokens", "stop_tokens_expanded"):
                raise ValueError("Unsupported line splitting mode.")
        if self.verify_schedule not in ("sequential", "parallel"):
            raise ValueError("Unsupported verification schedule.")
        if (self.max_slot_tokens is None) != (self.summary_token_budget is None):
            raise ValueError("Set both slot and summary token budgets, or neither.")
        if self.max_slot_tokens is not None:
            if (
                type(self.max_slot_tokens) is not int
                or type(self.summary_token_budget) is not int
                or self.max_slot_tokens < 1
                or self.summary_token_budget < 1
            ):
                raise ValueError("Token budgets must be positive integers.")
            if self.verify_lookahead != 1:
                raise ValueError("Token-bounded compaction requires verify_lookahead=1.")


@dataclass
class Data:
    model_name: str = "Qwen/Qwen3-0.6B"
    hf_dataset: str | None = None
    local_data_dir: str | None = None
    splits: list[str] = field(default_factory=lambda: ["tau3", "appworld", "liveclawbench"])
    output_dir: str = "data/slot-sft"
    eval_ratio: float = 0.05
    split_mode: Literal["hash", "shuffle"] = "hash"
    seed: int = 17
    cutoff_len: int = 16384
    context_floor: int = 256
    max_target_tokens: int = 1024
    max_examples_per_split: int = 0

    def validate(self):
        if bool(self.hf_dataset) == bool(self.local_data_dir):
            raise ValueError("Specify exactly one of hf_dataset and local_data_dir.")
        if not self.splits or any(not isinstance(s, str) or not s for s in self.splits):
            raise ValueError("splits must name at least one source split.")
        if not 0 <= self.eval_ratio < 1 or self.split_mode not in ("hash", "shuffle"):
            raise ValueError("Invalid held-out split configuration.")
        for name in ["cutoff_len", "context_floor", "max_target_tokens"]:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be positive.")
        if type(self.max_examples_per_split) is not int or self.max_examples_per_split < 0:
            raise ValueError("max_examples_per_split must be nonnegative.")


@dataclass
class Training:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    tokenized_path: str = "data/slot-sft"
    output_dir: str = "outputs/slot-sft"
    flash_attn: str = "fa3"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.05
    cutoff_len: int = 16384
    bf16: bool = True
    seed: int = 42
    dataloader_num_workers: int = 4
    report_to: str = "none"
    run_name: str = "slot-sft"
    logging_steps: int = 10
    eval_steps: int = 200
    per_device_eval_batch_size: int = 4
    disable_gradient_checkpointing: bool = True

    def validate(self):
        if not self.tokenized_path or not self.output_dir or not self.model_name_or_path:
            raise ValueError("Training model, dataset and output paths are required.")
        for name in [
            "learning_rate",
            "num_train_epochs",
            "warmup_ratio",
            "lora_dropout",
            "weight_decay",
            "max_grad_norm",
        ]:
            if type(getattr(self, name)) not in (int, float) or not math.isfinite(
                getattr(self, name)
            ):
                raise ValueError(name + " must be a finite number.")
        for name in ["bf16", "disable_gradient_checkpointing"]:
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + " must be boolean.")
        for name in [
            "lora_rank",
            "lora_alpha",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "cutoff_len",
            "logging_steps",
            "eval_steps",
            "per_device_eval_batch_size",
        ]:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer.")
        if (
            self.learning_rate <= 0
            or self.num_train_epochs <= 0
            or not 0 <= self.warmup_ratio < 1
            or not 0 <= self.lora_dropout < 1
        ):
            raise ValueError("Invalid optimizer or LoRA configuration.")


def load_config(path: Path, kind: Literal["compact", "prepare-sft", "train-sft"]):
    """Parse one YAML mapping, reject unknown fields, and validate before use."""
    values = yaml.safe_load(path.read_text(encoding="utf8"))
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    if kind == "compact":
        cls = Compaction
        values = dict(values)
        for role in ["draft", "target"]:
            if values.get(role) is not None:
                if not isinstance(values[role], dict):
                    raise ValueError(role + " must be an endpoint mapping.")
                values[role] = Endpoint(**values[role])
    elif kind == "prepare-sft":
        cls = Data
    elif kind == "train-sft":
        cls = Training
    else:
        raise ValueError("Unsupported operation: " + kind)
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
    result = cls(**values)
    result.validate()
    return result
