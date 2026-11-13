"""Control availability rules shared with the original Qt UI."""

import sys
from .schema import SPEC
from ..training.optimizer_specs import supports

# Controls that exist but would do nothing in the current combination. Each entry is
# key -> predicate(flat_config) -> enabled.
_RULES = {
    **{
        f"paired_texture.{option}": (
            lambda c: float(c.get("paired_texture.scale") or 0) > 0
        )
        for option in (
            "levels",
            "patch",
            "tokens",
            "features",
            "seed",
            "snr_cut",
            "min_gate",
        )
    },
    **{
        f"flow.{option}": (lambda c: float(c.get("flow.hf_scale") or 0) > 0)
        for option in (
            "hf_high_noise_snr_cut",
            "hf_high_noise_min_weight",
            "hf_high_noise_power",
        )
    },
    **{
        f"adversarial.{option}": (lambda c: bool(c.get("adversarial.enabled")))
        for option in ("weight", "discriminator_lr", "warm_start")
    },
    **{
        f"perceptual.{option}": (lambda c: bool(c.get("perceptual.enabled")))
        for option in (
            "weight",
            "max_t",
            "every",
            "cache_dir",
            "eye_manifest",
            "dedicated_low_noise",
        )
    },
    **{
        f"self_flow.{option}": (lambda c: bool(c.get("self_flow.enabled")))
        for option in (
            "ema_dtype",
            "ema_device",
            "adaln_fp32",
            "stochastic_rounding",
            "decay",
            "weight",
        )
    },
    # Measured: under `uniform`, scale 0.5/1.0/2.0 give byte-identical distributions.
    "flow.sigmoid_scale": lambda c: (
        c.get("flow.timestep_sample_method") == "logit_normal"
    ),
    "flow.shift": lambda c: c.get("flow.timestep_sample_method") != "moss",
    "flow.flux_shift": lambda c: c.get("flow.timestep_sample_method") != "moss",
    **{
        f"flow.moss_{option}": (
            lambda c: c.get("flow.timestep_sample_method") == "moss"
        )
        for option in ("mu", "std", "cutoff", "uniform_mix", "p_full", "steps")
    },
    "flow.dual_timestep_mask_ratio": lambda c: bool(c.get("flow.dual_timestep")),
    "flow.hf_exponent": lambda c: float(c.get("flow.hf_scale") or 0) > 0,
    **{
        f"rti.{option}": (lambda c: bool(c.get("rti.enabled")))
        for option in (
            "dense_prefix_blocks",
            "dense_suffix_blocks",
            "size_buckets",
            "start_keep",
            "target_keep",
            "identity_steps",
            "warmup_steps",
            "anneal_steps",
            "budget_steps",
        )
    },
    "quant.matmul_tuning": lambda c: (
        c.get("quant.preset") in ("balanced", "fast")
        or (
            c.get("quant.backend") == "sdnq"
            and c.get("quant.mode") == "frozen"
            and c.get("quant.use_quantized_matmul") is True
        )
    ),
    "train.epochs": lambda c: not c.get("resolution_schedule.enabled"),
    "train.save_every_epochs": lambda c: not c.get("resolution_schedule.enabled"),
    "dataset.tier_collapse": lambda c: (
        bool(c.get("dataset.resolutions")) and not c.get("resolution_schedule.enabled")
    ),
    "resolution_schedule.stages": lambda c: bool(c.get("resolution_schedule.enabled")),
    "resolution_schedule.save_stage_end": lambda c: bool(
        c.get("resolution_schedule.enabled")
    ),
    "train.resume_mode": lambda c: bool(c.get("train.resume_from")),
    "train.moss_qkv_format": lambda c: (
        c.get("train.model_family") in ("auto", "moss_image")
        and (
            c.get("adapter.kind") in ("none", "lora")
            or (
                c.get("adapter.kind") == "lycoris_lora"
                and c.get("adapter.lycoris_algo") == "locon"
            )
        )
    ),
    "train.activation_offload_budget_mb": lambda c: bool(
        c.get("train.activation_offload")
    ),
    "train.activation_offload_pin_memory": lambda c: bool(
        c.get("train.activation_offload")
    ),
    "train.activation_offload_prefetch": lambda c: (
        bool(c.get("train.activation_offload"))
        and bool(c.get("train.activation_offload_pin_memory"))
    ),
    "dataset.resolution": lambda c: not c.get("dataset.resolutions"),
    "dataset.min_source_area": lambda c: bool(c.get("dataset.resolutions")),
    # Inert on the no_upscale path -- verified, 256/128/64 give identical buckets.
    "dataset.min_bucket_reso": lambda c: not c.get("dataset.bucket_no_upscale", True),
    "dataset.caption.mixed_weights": lambda c: (
        c.get("dataset.caption.caption_mode") == "mixed"
    ),
    "dataset.caption.shuffle_keep_first_n": lambda c: bool(
        c.get("dataset.caption.shuffle_tags")
    ),
    "dataset.caption.nl_keep_first_sentence": lambda c: bool(
        c.get("dataset.caption.nl_shuffle_sentences")
    ),
    # Texture settings only mean anything if some phase asks for texture. There is no separate
    # "texture mode" switch to gate on -- the curriculum IS the switch -- so the rule reads the
    # phase list. `flow.phase_mapping` is inert without any curriculum at all, texture or not.
    "flow.phase_mapping": lambda c: bool(c.get("curriculum")),
    **{
        f"optimizer.{option}": (
            lambda c, option=option: supports(c.get("optimizer.kind"), option)
        )
        for option in (
            "betas",
            "eps",
            "use_kahan",
            "kahan_sum",
            "momentum",
            "quantize_state",
            "offload_state",
            "gradient_release",
            "norm_mode",
            "use_first_moment",
            "sinkhorn_iterations",
        )
    },
    "schedule.min_lr_ratio": lambda c: c.get("schedule.kind") != "constant",
    "schedule.d": lambda c: c.get("schedule.kind") == "rex",
    "schedule.global_d": lambda c: c.get("schedule.kind") == "rerex",
    "schedule.local_d": lambda c: c.get("schedule.kind") == "rerex",
    "schedule.weight_power": lambda c: c.get("schedule.kind") == "rerex",
    "schedule.num_segments": lambda c: c.get("schedule.kind") == "rerex",
    "train.text_cache_batch_size": lambda c: bool(c.get("train.cache_text_embeddings")),
    "train.compile_text_encoder": lambda c: (
        not bool(c.get("train.cache_text_embeddings"))
    ),
    "train.text_encoder_embedding_only": lambda c: (
        not bool(c.get("train.cache_text_embeddings"))
    ),
    "train.caption_variations": lambda c: bool(c.get("train.cache_text_embeddings")),
    "train.caption_cache_path": lambda c: (
        bool(c.get("train.cache_text_embeddings"))
        and bool(c.get("train.caption_variations"))
    ),
    "adapter.lycoris_algo": lambda c: c.get("adapter.kind") == "lycoris_lora",
    "adapter.lycoris_bypass": lambda c: (
        c.get("adapter.kind") == "lycoris_lora"
        and c.get("adapter.lycoris_algo") not in ("dora",)
    ),
    "adapter.lycoris_wd_on_output": lambda c: (
        c.get("adapter.kind") == "lycoris_lora"
        and c.get("adapter.lycoris_algo") in ("dora",)
    ),
    "adapter.dtype": lambda c: c.get("adapter.kind") != "none",
    "adapter.moss_train_text_refiner": lambda c: (
        c.get("train.model_family") in ("auto", "moss_image")
        and c.get("adapter.kind") != "none"
    ),
    "block_lr.enabled": lambda c: (
        bool(c.get("block_lr.enabled"))
        or (
            c.get("train.model_family") in ("auto", "moss_image")
            and (
                c.get("adapter.kind") == "lora"
                or (
                    c.get("adapter.kind") == "lycoris_lora"
                    and c.get("adapter.lycoris_algo") == "locon"
                )
            )
        )
    ),
    "block_lr.ranges": lambda c: bool(c.get("block_lr.enabled")),
    "adapter.rank": lambda c: c.get("adapter.kind") != "none",
    "adapter.alpha": lambda c: c.get("adapter.kind") != "none",
    "adapter.dropout": lambda c: c.get("adapter.kind") != "none",
    "adapter.components": lambda c: c.get("adapter.kind") != "none",
    "adapter.lokr_factor": lambda c: (
        c.get("adapter.kind") == "lokr"
        or (
            c.get("adapter.kind") == "lycoris_lora"
            and c.get("adapter.lycoris_algo") in ("lokr",)
        )
    ),
    "adapter.lokr_decompose_both": lambda c: (
        c.get("adapter.kind") == "lokr"
        or (
            c.get("adapter.kind") == "lycoris_lora"
            and c.get("adapter.lycoris_algo") in ("lokr",)
        )
    ),
    "train.compile_dynamic": lambda c: bool(c.get("train.compile")),
    "train.compile_regional": lambda c: bool(c.get("train.compile")),
    "train.pack_resolutions": lambda c: (
        sys.platform != "win32" or c.get("train.attention_backend") == "flash_attn_2"
    ),
    "train.moss_checkpoint_inside_compile": lambda c: (
        c.get("train.model_family") in ("auto", "moss_image")
        and bool(c.get("train.compile"))
        and bool(c.get("train.gradient_checkpointing"))
    ),
    "quant.text_encoder_weights_dtype": lambda c: bool(
        c.get("quant.quantize_text_encoder")
    ),
    "quant.weights_dtype": lambda c: (
        (c.get("quant.backend", "sdnq") == "sdnq" and c.get("quant.mode") != "none")
        or c.get("quant.quantize_text_encoder")
    ),
    "quant.backend": lambda c: True,
    "quant.mode": lambda c: True,
    "quant.use_quantized_matmul": lambda c: (
        c.get("quant.backend", "sdnq") == "sdnq" and c.get("quant.mode") != "none"
    ),
    "quant.use_hadamard": lambda c: (
        c.get("quant.backend", "sdnq") == "sdnq"
        and c.get("quant.mode") == "frozen"
        and c.get("quant.use_quantized_matmul") is True
    ),
    "quant.skip_policy": lambda c: c.get("quant.mode") != "none",
    "quant.extra_skip": lambda c: c.get("quant.mode") != "none",
    "quant.group_size": lambda c: (
        (c.get("quant.backend", "sdnq") == "sdnq" and c.get("quant.mode") != "none")
        or c.get("quant.quantize_text_encoder")
    ),
    "quant.dynamic_loss_threshold": lambda c: (
        (c.get("quant.backend", "sdnq") == "sdnq" and c.get("quant.mode") != "none")
        or c.get("quant.quantize_text_encoder")
    ),
    "quant.use_stochastic_rounding": lambda c: (
        c.get("quant.backend", "sdnq") == "sdnq" and c.get("quant.mode") == "training"
    ),
}


def _has_texture_phase(c: dict) -> bool:
    return any(p.get("mode") == "texture" for p in (c.get("curriculum") or []))


# Derived from the schema rather than listed, so adding a texture key cannot leave it ungated --
# which is exactly what happened when the oversize cascade added four of them at once.
for _tex in (k for k in SPEC if k.startswith("dataset.texture.")):
    _RULES[_tex] = lambda c: _has_texture_phase(c)


def _adapter_targets(c: dict) -> set[str]:
    t = set(c.get("adapter.components") or [])
    return t


for _comp in ("image_attn", "text_attn", "mlp", "adaln", "base"):
    # Under an adapter, a per-component LR only reaches parameters that actually have one injected
    # -- which is exactly what `load_config` rejects. Full FT can address every component.
    _RULES[f"component_lr.{_comp}"] = lambda c, comp=_comp: (
        c.get("adapter.kind") == "none" or comp in _adapter_targets(c)
    )


_RULES["component_lr.adaln"] = lambda c: c.get("adapter.kind") == "none"
