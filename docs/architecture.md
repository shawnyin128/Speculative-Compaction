# Core architecture

```text
src/
  entrypoint.py             # CLI: parse, validate, compose
  hparam/hparam.py          # typed configuration and validation
  compaction/
    compaction.py          # public Compactor, lifecycle and accounting
    slots.py               # slot-parallel draft/verify/repair algorithm
    verification.py        # semantic scores and deterministic gates
    clients.py             # model HTTP calls, line splitting and buffers
    prompts.py             # shared training/inference prompt contract
    *.txt                  # unchanged prompt templates
  data/data.py             # SFT DatasetDict construction
  data/encoding.py         # pure prompt/target encoding and masking
  training/training.py     # LLaMA-Factory SFT composition
  LlamaFactory/            # separately packaged local framework runtime
  utils/measurement_clock.py
configs/                   # experiment values, not a second schema
scripts/                   # thin shell entrypoints
tests/                     # core regression and configuration checks
asset/demo.gif
```

The dependency direction is entrypoint → capability composition → implementation.
Configuration owns no models, tensors, clients or framework imports. There is no
benchmark package dependency and no ambient `PYTHONPATH` requirement.

## Compaction

`Compactor.compact(context_text)` receives an already-rendered context and returns
`CompactionResult(text, metadata)`. It does not choose historical turns, trim the
context, or invoke an agent framework. Oversized model prompts raise explicitly.

Each call creates independent draft/repair clients and a shared verification
prefix. The draft and target use their own tokenizer templates. The six slot
loops can run serially or concurrently; accepted lines form each slot's actual
prefix. Rejected proposals are repaired by the target. Buffered draft lines are
reused only when the accepted prefix remains valid. Target repairs never reuse
unverified extra lines. Token budgets include repaired output.

The fixed slot prompts remain compatible with the existing SFT format. The
single-slot draft schema remains distinct from the full-schema target prefix.
The existing `dreamr` probability score and `sr` digit score are separate scales;
threshold equality accepts, and a missing answer label rejects safely.

Metadata separates source/summary length from total draft/judge/repair generation
work. Per-role times may overlap and must not be summed as compaction wall time.

## SFT

`build_sft_dataset` loads existing slot-target rows from local Parquet or the Hub,
splits by trace group, assembles the exact chat-template text, masks prompt labels
with -100, and preserves target EOS. Only overlong context tails may shrink;
an overlong supervised target is rejected, not truncated. Dataset caching state
is restored even if a build fails.

`train_sft` validates the pretokenized train/validation DatasetDict before importing
the optional trainer. It passes the explicit LoRA SFT recipe to the actual
LLaMA-Factory `run_exp`. The supplied dataset already owns its validation split;
the framework `val_size` setting is only a parser placeholder.

## Environments

The base/dev environment is lightweight. `inference` adds tokenizers and
`training` adds the local LLaMA-Factory runtime. These groups intentionally
conflict: the preserved trainer caps Transformers/Datasets below the serving
tokenizer environment. Do not combine groups with `--all-groups`.

LLaMA-Factory's internal packages are retained as framework source. They are not
project benchmark evaluation or native Hermes/OpenClaw integrations.
