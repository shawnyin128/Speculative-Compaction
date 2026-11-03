# Retained training runtime

This module preserves the locally used LLaMA-Factory package and its independent
pyproject, README, manifest and Apache-2.0 license. It is installed only with the
project's training dependency group.

The retained source includes the project's pretokenized-data and attention
compatibility work, including data/loader.py, data/mm_plugin.py,
model/model_utils/attention.py, hparams/finetuning_args.py and train/tuner.py.
No new trainer algorithm or upstream source-style rewrite was performed during
the core extraction. Existing removed experimental GRPO files remain absent.

Only framework runtime source and build metadata are included. Original clone
metadata, environment files, framework examples/assets, run outputs and private
machine configuration are not distributed.
