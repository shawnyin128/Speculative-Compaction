#!/usr/bin/env bash
set -euo pipefail
uv run --locked --group inference speccompact compact --config configs/compact/gemma4.yaml "$@"
