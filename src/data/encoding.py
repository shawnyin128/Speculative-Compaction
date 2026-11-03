"""Pure SFT row encoding: exact prompt, target EOS and prompt-only label mask."""

from collections import Counter

from compaction.prompts import build_shared_prefix, build_slot_suffix
from hparam.hparam import Data

IM_END = "<|im_end|>"
IGNORE_INDEX = -100


def encode_slot(row: dict, parameters: Data, tokenizer):
    """Return a training sample and build diagnostics without writing artifacts.

    Only context may be shortened. A supervised target is either kept in full
    with its EOS or rejected. Input and label vectors have the same length T;
    the first P prompt labels are -100 and the T-P target labels are token IDs.
    """
    counts = Counter()
    target_tail = row["slot_target"] + IM_END + "\n"

    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    if len(encode(row["slot_target"])) > parameters.max_target_tokens:
        return None, {"counts": {"dropped_long_target": 1}}
    context = row["context_text"]
    context_ids = encode(context)
    for _ in range(3):
        user = build_shared_prefix(context, slots=(row["slot"],)) + build_slot_suffix(row["slot"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = prompt + target_tail
        prompt_ids = encode(prompt)
        input_ids = encode(full_text)
        if len(input_ids) <= parameters.cutoff_len:
            break
        keep = len(context_ids) - (len(input_ids) - parameters.cutoff_len)
        if keep < parameters.context_floor:
            counts["dropped_context_floor"] += 1
            return None, {"counts": dict(counts)}
        context_ids = context_ids[:keep]
        context = tokenizer.decode(context_ids)
        counts["truncated"] += 1
    else:
        counts["dropped_still_long"] += 1
        return None, {"counts": dict(counts)}
    if input_ids[: len(prompt_ids)] != prompt_ids:
        counts["fallback_boundary"] += 1
        input_ids = prompt_ids + encode(target_tail)
    labels = [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    if len(labels) != len(input_ids) or input_ids[-2:] != encode(IM_END + "\n"):
        raise ValueError("SFT sample lost its aligned labels or final EOS/newline.")
    counts["kept"] += 1
    return {"input_ids": input_ids, "labels": labels, "from": row["from"]}, {
        "counts": dict(counts),
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(input_ids) - len(prompt_ids),
        "preview": full_text,
    }
