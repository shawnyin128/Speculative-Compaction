#!/usr/bin/env bash
set -euo pipefail
uv run --locked --group training speccompact prepare-sft --config configs/data/slot-sft.yaml "$@"
