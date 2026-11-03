# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...extras import logging
from ...extras.constants import AttentionFunction
from ...extras.packages import is_torch_version_greater_than


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def configure_attn_implementation(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    from transformers.utils import is_flash_attn_2_available

    if getattr(config, "model_type", None) == "gpt_oss":
        from transformers.integrations.hub_kernels import load_and_register_kernel

        flash_attn3_kernel = "kernels-community/vllm-flash-attn3"
        load_and_register_kernel(flash_attn3_kernel)
        setattr(config, "_attn_implementation", flash_attn3_kernel)
        setattr(config, "_attn_implementation_internal", flash_attn3_kernel)
        model_args.flash_attn = AttentionFunction.FA3

        logger.info_rank0("Using FlashAttention-3 with attention sink for the gpt-oss model.")
        return

    if getattr(config, "model_type", None) == "gemma2":
        if model_args.flash_attn == AttentionFunction.AUTO or model_args.flash_attn == AttentionFunction.FA2:
            if is_flash_attn_2_available():
                if model_args.flash_attn != AttentionFunction.FA2:
                    logger.warning_rank0("Gemma 2 should use flash attention 2, change `flash_attn` to fa2.")
                    model_args.flash_attn = AttentionFunction.FA2
            else:
                logger.warning_rank0("FlashAttention-2 is not installed, use eager attention.")
                model_args.flash_attn = AttentionFunction.DISABLED
        elif model_args.flash_attn == AttentionFunction.SDPA:
            logger.warning_rank0(
                "Gemma-2 should use soft-capping attention, while the SDPA attention does not support it."
            )

    if getattr(config, "model_type", None) in ["youtu", "youtu_vl"]:
        if model_args.flash_attn in (AttentionFunction.AUTO, AttentionFunction.SDPA):
            logger.warning_rank0("Youtu-VL does not support SDPA, forcing eager attention.")
            model_args.flash_attn = AttentionFunction.DISABLED

    if model_args.flash_attn == AttentionFunction.AUTO:
        return

    elif model_args.flash_attn == AttentionFunction.DISABLED:
        requested_attn_implementation = "eager"

    elif model_args.flash_attn == AttentionFunction.SDPA:
        if not is_torch_version_greater_than("2.1.1"):
            logger.warning_rank0("torch>=2.1.1 is required for SDPA attention.")
            return

        requested_attn_implementation = "sdpa"
    elif model_args.flash_attn == AttentionFunction.FA2:
        from transformers import is_torch_npu_available

        if not (is_flash_attn_2_available() or is_torch_npu_available()):
            logger.warning_rank0("FlashAttention-2 is not installed.")
            return

        requested_attn_implementation = "flash_attention_2"
    elif model_args.flash_attn == AttentionFunction.FA3:
        # Local patch (SpecCompact): generic FA3 via the prebuilt hub kernel,
        # the same no-compile route the gpt-oss branch above uses. Needs
        # `pip install kernels` and hub access on first load (cached after).
        # Falls back to the transformers default attention on any failure.
        try:
            from transformers.integrations import hub_kernels

            # The loader entrypoint is named differently across transformers
            # versions (load_and_register_attn_kernel in 5.8, plain
            # load_and_register_kernel elsewhere) -- take whichever exists.
            loader = getattr(hub_kernels, "load_and_register_attn_kernel", None) or getattr(
                hub_kernels, "load_and_register_kernel", None
            )
            if loader is None:
                raise ImportError("no kernel loader found in transformers.integrations.hub_kernels")

            # Newer `kernels` releases require an explicit revision, so try the
            # @main-pinned form first and fall back to the bare repo id. The
            # kernel registers under the exact string we pass, so the config's
            # attn implementation must use the same string.
            flash_attn3_kernel = "kernels-community/vllm-flash-attn3@main"
            try:
                loader(flash_attn3_kernel)
            except Exception:
                flash_attn3_kernel = "kernels-community/vllm-flash-attn3"
                loader(flash_attn3_kernel)

            setattr(config, "_attn_implementation", flash_attn3_kernel)
            setattr(config, "_attn_implementation_internal", flash_attn3_kernel)
            logger.info_rank0(f"Using FlashAttention-3 via hub kernel ({flash_attn3_kernel}).")
        except Exception as err:
            logger.warning_rank0(f"Failed to load the FA3 hub kernel ({err}); using default attention.")

        return
    else:
        raise NotImplementedError(f"Unknown attention type: {model_args.flash_attn}")

    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        setattr(config, "attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "kimi_vl":
        setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "youtu_vl":
        setattr(config, "attn_implementation", requested_attn_implementation)
        setattr(config, "_attn_implementation", requested_attn_implementation)
        if hasattr(config, "vision_config"):
            setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    else:
        setattr(config, "_attn_implementation", requested_attn_implementation)


def print_attn_implementation(config: "PretrainedConfig") -> None:
    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        attn_implementation = getattr(config, "attn_implementation", None)
    else:
        attn_implementation = getattr(config, "_attn_implementation", None)

    if attn_implementation == "flash_attention_2":
        logger.info_rank0("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info_rank0("Using torch SDPA for faster training and inference.")
    elif attn_implementation and "/" in attn_implementation:
        # Local patch (SpecCompact): hub-kernel implementations (e.g. FA3)
        # carry a repo-id string; they are not vanilla attention.
        logger.info_rank0(f"Using hub kernel attention implementation ({attn_implementation}).")
    else:
        logger.info_rank0("Using vanilla attention implementation.")
