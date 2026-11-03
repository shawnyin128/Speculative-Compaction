"""Build the original prompt-masked, trace-disjoint SFT DatasetDict."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from compaction.prompts import SLOT_CONTRACTS
from data.encoding import encode_slot
from hparam.hparam import Data


def percentiles(values: list[int]) -> dict[str, int]:
    """Return p50/p90/p99/max for a non-empty list of lengths."""
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {"p50": pick(0.50), "p90": pick(0.90), "p99": pick(0.99), "max": ordered[-1]}


def build_sft_dataset(parameters: Data, *, tokenizer=None):
    """Save complete train/validation data; never truncate a supervised target.

    Dependencies and tokenizer are loaded only for the requested build. Context
    trimming, trace-level hash/shuffle split, EOS and label masking preserve the
    existing training pipeline. Caller-global dataset caching is restored.
    """
    parameters.validate()
    output = Path(parameters.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new dataset directory: " + str(output))
    import transformers
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        Sequence,
        Value,
        disable_caching,
        enable_caching,
        is_caching_enabled,
        load_dataset,
    )
    from transformers import AutoTokenizer

    cached = is_caching_enabled()
    try:
        args = parameters
        split_names = tuple(args.splits)
        # from_generator caches by function hash; a cache hit would skip the generator and
        # leave the manifest stats empty, so a build must always regenerate for real.
        disable_caching()

        # Step 1: load source rows from local parquet or the hub, tagging provenance.
        rows: list[dict[str, str]] = []
        for split in split_names:
            if args.local_data_dir is not None:
                shards = sorted(str(p) for p in Path(args.local_data_dir).glob(f"{split}*.parquet"))
                if not shards:
                    raise FileNotFoundError(
                        f"[ERROR] no {split}*.parquet under {args.local_data_dir}"
                    )
                source = load_dataset("parquet", data_files={split: shards})[split]
            else:
                source = load_dataset(args.hf_dataset, split=split)
            if args.max_examples_per_split:
                source = source.select(range(min(args.max_examples_per_split, len(source))))
            kept = 0
            for item in source:
                if item["slot"] not in SLOT_CONTRACTS:
                    continue
                rows.append(
                    {
                        "trace_id": str(item["trace_id"]),
                        "slot": str(item["slot"]),
                        "context_text": str(item["context_text"]),
                        "slot_target": str(item["slot_target"]).strip(),
                        "from": split,
                    }
                )
                kept += 1
            print(f"[INFO] loaded split={split} rows={kept}")
        if not rows:
            raise ValueError("[ERROR] no rows loaded")

        # Step 2: hold out eval by trace group so no context spans both sides. Hash mode
        # keys the side on md5(trace_id) alone, making the split stable across sibling
        # datasets (e.g. different teachers over overlapping trace universes).
        groups = sorted({(row["from"], row["trace_id"]) for row in rows})
        if args.split_mode == "hash":
            permille = int(args.eval_ratio * 1000)

            def is_eval(trace_id: str) -> bool:
                digest = hashlib.md5(trace_id.encode("utf-8")).hexdigest()
                return int(digest, 16) % 1000 < permille

            eval_groups = {(source, trace_id) for source, trace_id in groups if is_eval(trace_id)}
        else:
            rng = random.Random(args.seed)
            rng.shuffle(groups)
            eval_count = max(1, round(len(groups) * args.eval_ratio)) if args.eval_ratio > 0 else 0
            eval_groups = set(groups[:eval_count])
        train_rows = [row for row in rows if (row["from"], row["trace_id"]) not in eval_groups]
        eval_rows = [row for row in rows if (row["from"], row["trace_id"]) in eval_groups]
        print(
            f"[INFO] {args.split_mode} split: {len(groups)} traces -> {len(train_rows)} train rows, "
            f"{len(eval_rows)} eval rows ({len(eval_groups)} eval traces)"
        )

        # Step 3: tokenizer + per-row conversion: final chat-template text first, then one
        # tokenize pass; boundary asserted against the separately tokenized prompt text.
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if not tokenizer.chat_template:
            raise ValueError(f"[ERROR] {args.model_name} exposes no chat template")
        stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        prompt_lens: dict[str, list[int]] = defaultdict(list)
        target_lens: dict[str, list[int]] = defaultdict(list)
        previews: list[str] = []

        def convert(row):
            sample, diagnostics = encode_slot(row, parameters, tokenizer)
            source = row["from"]
            for key, value in diagnostics["counts"].items():
                stats[source][key] += value
            if sample is not None:
                prompt_lens[source].append(diagnostics["prompt_tokens"])
                target_lens[source].append(diagnostics["target_tokens"])
                if len(previews) < 3:
                    previews.append(diagnostics["preview"])
            return sample

        # Step 4: stream both partitions through Dataset.from_generator so 58k long rows
        # never sit in Python lists (arrow-backed, constant memory).
        features = Features(
            {
                "input_ids": Sequence(Value("int32")),
                "labels": Sequence(Value("int32")),
                "from": Value("string"),
            }
        )

        # from_generator shards list-valued gen_kwargs and re-invokes the generator per
        # shard, so per-call enumerate cannot show progress; count in the enclosing scope.
        progress: dict[str, int] = defaultdict(int)

        def generate(partition: list[dict[str, str]], label: str, total: int):
            for row in partition:
                progress[label] += 1
                print(f"\r[INFO] {label}: {progress[label]}/{total}", end="", flush=True)
                sample = convert(row)
                if sample is not None:
                    yield sample

        dataset = DatasetDict()
        dataset["train"] = Dataset.from_generator(
            generate,
            features=features,
            gen_kwargs={"partition": train_rows, "label": "train", "total": len(train_rows)},
        )
        print()
        if eval_rows:
            dataset["validation"] = Dataset.from_generator(
                generate,
                features=features,
                gen_kwargs={"partition": eval_rows, "label": "validation", "total": len(eval_rows)},
            )
            print()

        # Step 5: save the DatasetDict (LLaMA-Factory tokenized_path contract) plus a
        # build manifest and a few full-text previews for human audit.
        Path(args.output_dir).parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(args.output_dir))
        manifest = {
            "hf_dataset": args.hf_dataset,
            "splits": list(split_names),
            "model_name": args.model_name,
            "eval_ratio": args.eval_ratio,
            "split_mode": args.split_mode,
            "seed": args.seed,
            "cutoff_len": args.cutoff_len,
            "context_floor": args.context_floor,
            "max_target_tokens": args.max_target_tokens,
            "rows": {name: len(split) for name, split in dataset.items()},
            "per_source": {
                source: {
                    **stats[source],
                    "prompt_tokens": percentiles(prompt_lens[source])
                    if prompt_lens[source]
                    else None,
                    "target_tokens": percentiles(target_lens[source])
                    if target_lens[source]
                    else None,
                }
                for source in split_names
                if stats[source]
            },
            "transformers_version": transformers.__version__,
            "chat_template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode("utf-8")
            ).hexdigest(),
        }
        (Path(args.output_dir) / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (Path(args.output_dir) / "sample_previews.txt").write_text(
            "\n\n========\n\n".join(previews), encoding="utf-8"
        )
        print(f"[INFO] saved -> {args.output_dir}")
        print(json.dumps(manifest["per_source"], indent=2, ensure_ascii=False))
        return manifest
    finally:
        if cached:
            enable_caching()
