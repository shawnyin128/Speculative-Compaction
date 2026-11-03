#!/usr/bin/env bash
set -euo pipefail
uv run --locked --group training speccompact train-sft --config configs/train/gemma4.yaml "$@"
