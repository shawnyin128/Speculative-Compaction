# Core migration record

## Scope

This release extracts the maintained method from the research workspace into the
independent Speculative-Compaction repository. It includes core inference, slot
SFT dataset preparation, SFT training, presets, tests and the existing demo.
Benchmark evaluation, quality/system reports, native Hermes/OpenClaw patches,
participant data, dashboard code, Slurm scripts, machine-specific CUDA helpers,
old experimental trainers, model weights and run archives are not copied.

## Ownership changes

| Research implementation | Core release |
|---|---|
| SpecCompactMethod orchestration inside eval/method.py | compaction/compaction.py |
| model/spec_compactor.py | compaction/slots.py |
| model/spec_clients.py | compaction/clients.py |
| model/line_verifier.py | compaction/verification.py |
| prompt/slot_generation_prompts.py and shared templates | compaction/prompts.py and templates |
| training slot-SFT DatasetDict builder | data/data.py |
| per-teacher LoRA SFT launch configuration | hparam + configs/train + training/training.py |
| active local training/LlamaFactory runtime | src/LlamaFactory |

The new orchestration consumes typed attributes rather than repeatedly decoding
loosely typed dictionaries and method aliases. Public clients receive explicit
settings; the parameter owner is hparam. Framework and benchmark dispatch are
removed from the maintained inference path.

## Explicit interface changes

- The CLI accepts raw context text, not AppWorld/tau conversation objects.
- User configuration is YAML with strict field validation. Historical evaluation
  flags and aliases are not a compatibility interface in this core release.
- Defaults select the final optimized recipe: six slot workers, buffered client
  line splitting, shared target verification/repair prefix, structural bypass,
  28k prompt guard, 320-token slot budget and 2048-token assembled-summary budget.
  The draft keeps its original single-slot schema. Serial execution and other
  existing mathematical branches remain explicit parameter choices.
- Explicit eta=0 is now respected. The old dictionary wrapper used Python `or`
  and substituted a default. This is an intentional configuration correction,
  not a change to the verifier formula or the reported eta0.5/0.6/0.7 results.
- Training logging defaults to `none`; changing logging does not change the
  retained model, LoRA, optimizer, seed, length or dataset settings.
- Existing result and dataset directories are protected from accidental overwrite.

## Validation

The migrated regression suite checks line acceptance/rejection, repairs, buffered
prefix invalidation, tokenizer-specific newline stops, prompt guards, concurrency,
parallel fallback scheduling, threshold boundaries and usage accounting. New
checks cover typed configuration, the complete public composition, strict input
validation and installed CLI/import behavior outside the source tree.

An independent source comparison ran five complete old/new compaction paths with
fixed deterministic model responses: serial, parallel, draft-only, token-budgeted
and lookahead. Summary text, per-slot decision events, source/summary token counts,
acceptance/repair counts and per-role token accounting matched exactly.

Two complete old/new SFT data builds used actual local Parquet and Hugging Face
DatasetDict serialization, with a controlled tokenizer and no network. Both
trace-hash and seeded-shuffle modes produced identical partition membership,
input_ids, masked labels and EOS. The new build also restored dataset caching.

The core suite passes 96 CPU checks, including target-preserving SFT encoding
and the training adapter's input/output contract. Ruff lint and formatting pass.
Both the core and the separately packaged LLaMA-Factory runtime build as wheels.
The core wheel was installed noneditably in a second environment: imports,
packaged prompt assets, all five YAML presets and the CLI worked outside the
checkout without importing model or training frameworks. Vendored Python source
also passed syntax compilation.

No real model inference, CUDA execution or SFT training was run for this cleanup.
GPU numerical behavior, engine performance and the full trainer stack are not
claimed verified by these CPU checks. Existing paper results remain in the
original workspace and were not recomputed or changed.

Review was a source self-review and independent arithmetic/workflow checks, not a
separate agent review. An initial test invocation selected the parent research
directory; that command-selection error was corrected. Release checks run from
this repository's own directory and isolated installed environment.
