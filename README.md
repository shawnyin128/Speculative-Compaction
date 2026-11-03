<h1 align="center">SpecCompact: Accelerating LLM Agent Context Compaction with Slot-Parallel Semantic Speculative Verification</h1>

![SpecCompact demo](asset/demo.gif)

SpecCompact accelerates agent context compaction by drafting six semantic slots
in parallel. A target model verifies proposed lines and repairs rejected ones.
The result is a structured summary for continuing the conversation.

This repository contains the core method, SFT data preparation and training.

## Installation

Use Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:shawnyin128/Speculative-Compaction.git
cd Speculative-Compaction
uv sync --locked --group inference
uv run --locked --group inference speccompact --help
```

Inference and training use separate dependency groups because the retained
trainer and inference tokenizers require different library versions. Select one
group at a time; do not use `--all-groups`.

## Compact a context

Start OpenAI-compatible draft and target model servers separately. The draft
server must expose `/v1/completions`; the target must also expose
`/v1/chat/completions` with token log probabilities for verification. This package
does not launch or allocate model servers.

Edit `configs/compact/gemma4.yaml` or `configs/compact/qwen35.yaml` to match your
server URLs and served model names. Keep each endpoint's tokenizer matched to
that model. The presets use these model/tokenizer IDs:

| Preset | SFT drafter | Target |
|---|---|---|
| `gemma4` | `sy128/SpecComp-slot-sft-gemma4-31B-it` | `google/gemma-4-31B-it` |
| `qwen35` | `sy128/SpecComp-slot-sft-qwen3-5-122B-A10B` | `Qwen/Qwen3.5-122B-A10B` |

Save the rendered conversation as UTF-8 text, then run:

```bash
uv run --locked --group inference speccompact compact \
  --config configs/compact/gemma4.yaml \
  --input context.txt --output outputs/compacted.json
```

The JSON contains `text`, `metadata` and the effective `configuration`. Metadata
records wall time, acceptance and repair counts, per-role token usage, and timing.
Set `record_llm_calls: true` for individual call and slot decision records. Output
paths must be new. Input context is not silently truncated.

For authenticated servers, set the environment variable named by each endpoint's
`api_key_env` (default `VLLM_API_KEY`). Store only the variable name in YAML.

```python
from pathlib import Path

from compaction.compaction import Compactor
from hparam.hparam import load_config

parameters = load_config(Path("configs/compact/gemma4.yaml"), "compact")
compactor = Compactor(parameters)
result = compactor.compact(Path("context.txt").read_text(encoding="utf8"))
print(result.text)
```

Settings are defined and validated in
[`src/hparam/hparam.py`](src/hparam/hparam.py). Use `--print-config` to inspect a
preset without loading tokenizers or making model calls. Set
`slot_concurrency: 1` for serial slots; the default is six. `eta` controls the
semantic acceptance threshold on the selected verifier's scale.

## Prepare SFT data and train

The locally used LLaMA-Factory runtime is a separate package under
[`src/LlamaFactory`](src/LlamaFactory), with its own license and
[local change record](src/LlamaFactory/LOCAL_CHANGES.md).

```bash
uv sync --locked --group training
uv run --locked --group training speccompact prepare-sft \
  --config configs/data/slot-sft.yaml
uv run --locked --group training speccompact train-sft \
  --config configs/train/gemma4.yaml
```

Before running, edit the data and training presets to agree on paths. Source rows
require `trace_id`, `slot`, `context_text` and `slot_target`. Choose a Hub dataset
or a local directory containing `<split>*.parquet` files. The builder preserves
the training prompt contract, masks prompt labels, keeps the full target and EOS,
and writes an Arrow-backed `DatasetDict` with a build manifest.

The training command requires prebuilt `train` and `validation` partitions. It
invokes the real LLaMA-Factory LoRA SFT trainer; it does not generate teacher
targets. The default training model is `Qwen/Qwen3-0.6B`; teacher-specific presets
change the target dataset and output paths. GPU training requires a compatible
PyTorch/CUDA and attention setup; `flash_attn` is configurable in the training
preset.

## Development

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

See [architecture](docs/architecture.md) for module responsibilities and the
[migration record](docs/migration.md) for preserved behavior and validation limits.
The regression suite uses deterministic responses; it does not run GPU models.
