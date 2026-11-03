# SpecCompact core code

This repository contains compaction, SFT data preparation and training only.
Keep benchmark evaluation, native agent patches, study servers and cluster
orchestration in their original research workspace.

- Follow `docs/architecture.md` for responsibilities and public entrypoints.
- Own all user parameter definitions/defaults/validation in `src/hparam/`.
- Parse configuration once at `src/entrypoint.py`; keep algorithms in cohesive
  implementation files, with a same-named composition entry for each capability.
- Preserve prompt bytes, target/draft tokenizer separation, acceptance threshold
  semantics, accepted-prefix order, EOS/label masking and trace-level data splits.
- Imports and help must not load models, download data or allocate accelerators.
- Treat `src/LlamaFactory/` as a separately packaged upstream dependency. Preserve
  its license and local training patches; do not apply project style rewrites to it.
- Use the locked uv environment. Run `uv run --locked ruff check .`,
  `uv run --locked ruff format --check .` and `uv run --locked pytest`.
- Keep data, weights, credentials and generated run outputs outside version control.
- Never treat a mocked-model test as real GPU generation or training validation.
