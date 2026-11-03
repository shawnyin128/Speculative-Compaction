# LLaMA-Factory training module

This directory contains the locally used
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) runtime, distributed under
its original [Apache-2.0 license](LICENSE).

Install it through the SpecCompact training environment from the repository root:

```shell
uv sync --locked --group training
uv run --locked --group training speccompact train-sft --config configs/train/gemma4.yaml
```

Prepare the tokenized train/validation dataset as described in the repository
README before training. The retained local changes cover pretokenized loading
and attention compatibility in `data/loader.py`, `data/mm_plugin.py`,
`model/model_utils/attention.py`, `hparams/finetuning_args.py` and `train/tuner.py`.
