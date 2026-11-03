"""Prompt templates for shared-prefix slot generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SLOTS = (
    "goal",
    "constraints_preferences",
    "progress",
    "key_decisions",
    "next_steps",
    "critical_context",
)
SLOT_TITLES = {
    "goal": "Goal",
    "constraints_preferences": "Constraints & Preferences",
    "progress": "Progress",
    "key_decisions": "Key Decisions",
    "next_steps": "Next Steps",
    "critical_context": "Critical Context",
}

SLOT_CONTRACTS = {
    "goal": "What is the user trying to accomplish? Can be multiple items if the session covers different tasks.",
    "constraints_preferences": "Any constraints, preferences, or requirements mentioned by the user. Return (none) if none were mentioned.",
    "progress": "Use the OpenClaw Progress substructure: Done contains completed tasks/changes; In Progress contains current work; Blocked contains issues preventing progress, if any.",
    "key_decisions": "Decisions with brief rationale, using the pattern **Decision**: rationale.",
    "next_steps": "Ordered list of what should happen next.",
    "critical_context": "Any data, examples, or references needed to continue. Return (none) if not applicable. Preserve exact file paths, function names, and error messages.",
}

SLOT_FORMATS = {
    "goal": "[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]",
    "constraints_preferences": '- [Any constraints, preferences, or requirements mentioned by user]\n- [Or "(none)" if none were mentioned]',
    "progress": "### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Current work]\n\n### Blocked\n- [Issues preventing progress, if any]",
    "key_decisions": "- **[Decision]**: [Brief rationale]",
    "next_steps": "1. [Ordered list of what should happen next]",
    "critical_context": '- [Any data, examples, or references needed to continue]\n- [Or "(none)" if not applicable]',
}

TEMPLATE_DIR = Path(__file__).resolve().parent
SHARED_PREFIX_TEMPLATE = (TEMPLATE_DIR / "slot_generation_shared_prefix.txt").read_text(
    encoding="utf-8"
)
SLOT_SUFFIX_TEMPLATE = (TEMPLATE_DIR / "slot_generation_suffix.txt").read_text(encoding="utf-8")

DRAFT_SYSTEM_PROMPT = (
    "You compact an agent history into a single summary slot. "
    "Output only the slot content: no preamble, no JSON, no code fences, no repetition."
)


def render_draft_prompt(
    tokenizer: Any, user_prompt: str, *, system_prompt: str | None = None
) -> str:
    """Render the draft-generation prompt through the model's chat template.

    Instruction-tuned generators (e.g. Qwen3) only follow a slot-fill request
    when the chat template is applied; feeding the raw string makes them echo
    the instructions instead. Measured on Qwen3-0.6B with real histories: the
    raw prompt ran to a 1024-token cap without ever emitting EOS (3/3), while
    this rendering terminated 3/3. Falls back to the raw prompt for tokenizers
    that expose no chat template. Qwen3 "thinking" mode is disabled so the whole
    generation budget is spent on the slot, not a hidden reasoning trace.

    ``system_prompt`` defaults to None -- i.e. a bare user turn -- because that is
    the format the slot-SFT pipeline (``build_llamafactory_slot_sft.py``)
    teacher-forces, and the draft must be prompted exactly as it was trained or
    it sees a turn layout it never learned. Pass ``DRAFT_SYSTEM_PROMPT`` only if
    the SFT side adds one too.

    Args:
        tokenizer: Draft tokenizer; may or may not define a chat template.
        user_prompt: The slot-fill request (prefix + history + slot suffix).
        system_prompt: Optional system turn to prepend. None (default) renders a
            bare user turn, matching slot SFT.

    Returns:
        The chat-template-rendered prompt, or ``user_prompt`` unchanged when the
        tokenizer has no chat template.
    """
    if not getattr(tokenizer, "chat_template", None):
        return user_prompt
    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # Tokenizer template does not accept enable_thinking; render without it.
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# Step 1: build the shared trajectory prefix once for either parallel or one-shot slots.
def build_shared_prefix(trajectory_text: str, slots: tuple[str, ...] = SLOTS) -> str:
    slot_lines = "\n".join(f"- {slot}: {SLOT_CONTRACTS[slot]}" for slot in slots)
    return SHARED_PREFIX_TEMPLATE.format(
        slot_lines=slot_lines,
        trajectory=trajectory_text.strip(),
    )


# Step 2: build the per-slot suffix used by parallel slot generation.
def build_slot_suffix(slot: str) -> str:
    if slot not in SLOT_CONTRACTS:
        raise ValueError(f"Unknown slot: {slot}")
    return SLOT_SUFFIX_TEMPLATE.format(
        slot=slot,
        slot_title=SLOT_TITLES[slot],
        slot_contract=SLOT_CONTRACTS[slot],
        slot_format=SLOT_FORMATS[slot],
    )
