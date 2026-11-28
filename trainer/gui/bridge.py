"""TOML <-> GUI bridge.

Ours, not Aozora's. Two decisions carry the whole module.

**Widgets are keyed by dotted TOML path** (`flow.hf_scale`, `dataset.caption.tag_dropout_percent`)
rather than by an invented flat name space. There is then no rename table to drift: a key either
exists in the dataclasses or it does not, and `missing_ui_keys()` turns that into an assertion the
gate can run.

**Only non-default values are written.** Defaults come from the dataclasses themselves, so a
GUI-written TOML reads like a hand-written one -- and, more importantly, a field whose default
changes later is not silently pinned to the old value by every config the GUI ever saved.

Everything is validated by `trainer.training.config.load_config`, the same function the trainer
calls. The GUI does not reimplement a single rule; it shows you the error the trainer would raise.
"""

from __future__ import annotations

import dataclasses as dc
import tempfile
import tomllib
from pathlib import Path

import toml

from ..training.resolution_schedule import ResolutionSchedule
from ..training.block_lr import BlockLRConfig
from ..training.optimizer_specs import OPTIMIZERS, optimizer_defaults
from ..training.slider import SliderConfig

from ..data.caption import CAPTION_VARIANTS, MIX_DEFAULTS, CaptionConfig, compatible_mix
from ..data.texture import TextureConfig
from ..data.dataset import DatasetConfig, SubsetConfig
from ..training.config import (
    AdapterConfig,
    ComponentLRs,
    FlowConfig,
    OptimizerConfig,
    PreserveConfig,
    QuantConfig,
    RTIConfig,
    SelfFlowConfig,
    PerceptualConfig,
    EyeAdversarialConfig,
    PairedTextureConfig,
    ScheduleConfig,
    TrainConfig,
    load_config,
)

# TOML section -> dataclass. `dataset.caption` is nested one level deeper in the file than it is in
# the dataclass tree, which is the one place the two shapes disagree.
SECTIONS: dict[str, type] = {
    "resolution_schedule": ResolutionSchedule,
    "train": TrainConfig,
    "dataset": DatasetConfig,
    "dataset.caption": CaptionConfig,
    "dataset.texture": TextureConfig,
    "flow": FlowConfig,
    "optimizer": OptimizerConfig,
    "schedule": ScheduleConfig,
    "adapter": AdapterConfig,
    "preserve": PreserveConfig,
    "quant": QuantConfig,
    "component_lr": ComponentLRs,
    "block_lr": BlockLRConfig,
    "rti": RTIConfig,
    "self_flow": SelfFlowConfig,
    "perceptual": PerceptualConfig,
    "adversarial": EyeAdversarialConfig,
    "paired_texture": PairedTextureConfig,
    "slider": SliderConfig,
}

# Fields that exist on a dataclass but are not config keys in their own section -- they are
# nested tables with their own entry in SECTIONS, so listing them here too would demand a widget
# for the container itself.
_SKIP = {("dataset", "caption"), ("dataset", "texture")}

# `[[curriculum]]` is an array of tables, not a section of scalars, so it has no dataclass field to
# key a widget off. It is still a real config key with a real widget, so it is carried alongside
# the schema rather than special-cased at each call site. The value is a list of phase dicts; the
# empty list is the default and means "no curriculum", which is exactly today's behaviour.
ARRAY_KEYS: dict[str, list] = {"curriculum": []}


def dataset_paths(flat: dict) -> list[str]:
    """Active source folders for both GUIs' cache/audit jobs, in config order.

    Incomplete editor rows are ignored rather than treating an empty path as the
    repository root. Dataset validation still diagnoses duplicate source rows.
    """
    rows = flat.get("dataset.subsets") or [{
        "path": flat.get("dataset.path"),
        "num_repeats": flat.get("dataset.num_repeats", 1),
        "target_images": flat.get("dataset.target_images"),
    }]
    paths = []
    for row in rows:
        path = str(row.get("path") or "").strip()
        if not path:
            continue
        subset = SubsetConfig(path, num_repeats=row.get("num_repeats", 1),
                              target_images=row.get("target_images"))
        if subset.active and path not in paths:
            paths.append(path)
    return paths


def schema() -> dict[str, dc.Field]:
    """Every settable dotted key -> its dataclass field."""
    out: dict[str, dc.Field] = {}
    for section, klass in SECTIONS.items():
        for f in dc.fields(klass):
            if (section, f.name) in _SKIP:
                continue
            out[f"{section}.{f.name}"] = f
    return out


# Advanced TOML settings intentionally omitted from the Mage-Flow GUI.
CLI_ONLY_KEYS = {"adapter.components", "adapter.init", "adapter.init_cache", "dataset.caption.protected_tags_file"} | {
    key for key in schema()
    if key.startswith(("preserve.", "component_lr.")) and key != "component_lr.adaln"
}


def default_of(f: dc.Field):
    if f.default is not dc.MISSING:
        return f.default
    if f.default_factory is not dc.MISSING:       # type: ignore[misc]
        return f.default_factory()                # type: ignore[misc]
    return None


def defaults(optimizer_kind="adamw") -> dict:
    result = {**{k: default_of(f) for k, f in schema().items()},
              **{k: list(v) for k, v in ARRAY_KEYS.items()}}
    result["dataset.caption.mixed_weights"] = MIX_DEFAULTS["mixed"].copy()
    if optimizer_kind in OPTIMIZERS:
        result.update({f"optimizer.{k}": v for k, v in optimizer_defaults(optimizer_kind).items()})
    return result


# ---------------------------------------------------------------- nested <-> flat

# Form state only. Standalone modes in older TOMLs have their own missing-sidecar
# fallbacks and cache identities, so preserve them when unrelated settings are edited.
IMPORTED_CAPTION = "_imported_caption"
_MODE = "dataset.caption.caption_mode"
_WEIGHTS = "dataset.caption.mixed_weights"


def flatten(nested: dict) -> dict:
    """Nested TOML dict -> {dotted key: value}, for the sections we know about."""
    flat = {}
    for section in SECTIONS:
        node = nested
        for part in section.split("."):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if (section, key) in _SKIP:
                continue
            dotted = f"{section}.{key}"
            # TOML bare keys are strings, so `batch_size = { 512 = 4 }` reads back with string
            # keys while the widget and `TrainConfig.__post_init__` both use ints. Canonicalise
            # here so the flat dict has one key type and comparisons mean something.
            if dotted == "train.batch_size" and isinstance(value, dict):
                value = {int(k): int(v) for k, v in value.items()}
            if dotted == "adapter.lycoris_algo" and value == "lora":
                value = "locon"
            if dotted == "train.compile" and value is False:
                value = None
            if dotted == "train.save_every_epochs" and value == 0:
                value = None
            flat[dotted] = value
    caption_raw = nested.get("dataset", {}).get("caption", {})
    if isinstance(caption_raw, dict) and caption_raw.get("protected_tags_file") is not None:
        flat["dataset.caption.protected_tags_file"] = caption_raw["protected_tags_file"]
    for key in ARRAY_KEYS:
        value = nested.get(key)
        if isinstance(value, dict) and isinstance(value.get("phases"), list):
            value = value["phases"]
        if isinstance(value, list):
            flat[key] = [dict(entry) for entry in value if isinstance(entry, dict)]
    mode, weights = flat.get(_MODE), flat.get(_WEIGHTS)
    if mode in CAPTION_VARIANTS and mode not in ("tags", "nl"):
        flat[IMPORTED_CAPTION] = {"mode": mode, "weights": weights}
        flat[_WEIGHTS] = {mode: 100}
        flat[_MODE] = compatible_mix("mixed", flat[_WEIGHTS])
    elif mode in MIX_DEFAULTS:
        mode = compatible_mix(mode, weights)
        flat[_MODE] = mode
        flat[_WEIGHTS] = (MIX_DEFAULTS[mode].copy() if weights is None else
                          {k: v for k, v in weights.items() if v or k in MIX_DEFAULTS[mode]})
        if mode == "mixed_custom":
            flat[IMPORTED_CAPTION] = {"mode": mode}
            flat[_MODE] = "mixed"
    return flat


def _restore_imported_caption(flat: dict) -> dict:
    if IMPORTED_CAPTION not in flat:
        return flat
    flat = flat.copy()
    imported = flat.pop(IMPORTED_CAPTION)
    original = imported["mode"]
    if original == "mixed_custom":
        if flat.get(_MODE) == "mixed":
            flat[_MODE] = original
    elif flat.get(_MODE) == compatible_mix("mixed", {original: 100}):
        active = {k: weight for k, weight in flat[_WEIGHTS].items() if weight}
        if active == {original: 100}:
            flat[_MODE] = original
            flat[_WEIGHTS] = imported["weights"]
    return flat


def nest(flat: dict) -> dict:
    """{dotted key: value} -> nested TOML dict. Keys mapped to None are dropped, which is how the
    GUI spells "leave this unset" for the `int | None` and `float | None` fields."""
    out: dict = {}
    for key, value in _restore_imported_caption(flat).items():
        if value is None:
            continue
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def prune_defaults(flat: dict) -> dict:
    """Drop every value equal to its dataclass default. `dataset.path` and `train.run_name` are
    kept regardless -- a config without them is not obviously a config."""
    flat = _restore_imported_caption(flat)
    # Always write adapter.kind: it distinguishes full finetuning from adapter training.
    keep = {"dataset.path", "train.run_name", "train.model_path", "train.output_dir",
            "adapter.kind"}
    base = defaults(flat.get("optimizer.kind", "adamw"))
    mode = flat.get("dataset.caption.caption_mode")
    if mode in MIX_DEFAULTS:
        base["dataset.caption.mixed_weights"] = MIX_DEFAULTS[mode]
    out = {}
    for key, value in flat.items():
        if value is None:
            continue
        if key in keep or key not in base:
            out[key] = value
            continue
        d = base[key]
        # tuple/set/list mismatch: `betas` / `protected_tags` round-trip through TOML as lists.
        if isinstance(d, tuple) and isinstance(value, list):
            d = list(d)
        elif isinstance(d, set) and isinstance(value, list):
            d = sorted(d)
            value_cmp = sorted(value)
            if value_cmp != d:
                out[key] = value
            continue
        if value != d:
            out[key] = value
    if (
        "flow.shift" in flat
        and flat["flow.shift"] is None
        and not flat.get("flow.flux_shift")
        and not flat.get("flow.cabal_shift")
        and not flat.get("flow.res_shift")
        and flat.get("flow.timestep_sample_method") != "moss"
    ):
        out["flow.shift"] = 1.0
    if "train.save_every_epochs" in flat and flat["train.save_every_epochs"] is None:
        out["train.save_every_epochs"] = 0
    # `load_config` rejects a file that sets both. The GUI writes the file, so the right behaviour
    # is to emit the one that is in effect rather than to hand the user an error about a key they
    # cannot see. A multi-tier ladder wins; `resolution` is what "one tier" means.
    if out.get("dataset.resolutions"):
        out.pop("dataset.resolution", None)
    else:
        out.pop("dataset.resolutions", None)
    # Same rule, same reason: `path` and `subsets` are mutually exclusive in the loader, and the
    # GUI keeps a value in the path box even after subsets are added. Emit the one in effect.
    if out.get("dataset.subsets"):
        out.pop("dataset.path", None)
        out.pop("dataset.mask_path", None)  # subsets carry their own mask folders
        out.pop('dataset.num_repeats', None)
        out.pop('dataset.target_images', None)
        out['dataset.subsets'] = [{k: v for k, v in s.items() if v is not None
                                  and not (k == 'mask_path' and not v)
                                  and not (k == 'num_repeats' and s.get('target_images') is not None)}
                                 for s in out['dataset.subsets']]
    else:
        out.pop("dataset.subsets", None)
    if flat.get('resolution_schedule.enabled'):
        if flat.get('resolution_schedule.unit', 'steps') == 'steps':
            out.pop('train.epochs', None)
            out.pop('train.save_every_epochs', None)
        else:
            out.pop('train.max_steps', None)
            if flat.get('resolution_schedule.unit') == 'stage_epochs':
                out.pop('train.epochs', None)
    else:
        for key in (
            'resolution_schedule.stages',
            'resolution_schedule.unit',
            'resolution_schedule.save_stage_end',
            'resolution_schedule.restart_lr_each_stage',
        ):
            out.pop(key, None)
    if not flat.get("train.pack_resolutions") or flat.get("resolution_schedule.enabled"):
        out.pop("train.pack_mode", None)
        out.pop("train.pack_token_budget", None)
    elif flat.get("train.pack_mode", "samples") != "tokens":
        out.pop("train.pack_token_budget", None)
    if not flat.get("train.resume_from"):
        out.pop("train.resume_mode", None)
    if not flat.get("train.cache_text_embeddings"):
        out.pop("train.caption_variations", None)
        out.pop("train.caption_cache_path", None)
    else:
        out.pop("train.compile_text_encoder", None)
        out.pop("train.text_encoder_embedding_only", None)
    if flat.get("adapter.kind", "none") != "none":
        out.pop("component_lr.adaln", None)
    if flat.get("adapter.kind") != "lora":
        out.pop("adapter.loraplus_lr_ratio", None)
    if not (flat.get("train.activation_offload") and flat.get("train.activation_offload_pin_memory")):
        out.pop("train.activation_offload_prefetch", None)
    if not (
        flat.get("quant.backend", "sdnq") == "sdnq"
        and flat.get("quant.mode") in ("frozen", "training")
        and flat.get("quant.use_quantized_matmul") is True
    ):
        out.pop("quant.use_hadamard", None)
    if not (
        flat.get("train.model_family") in ("auto", "moss_image")
        and (
            flat.get("adapter.kind") in ("none", "lora")
            or (
                flat.get("adapter.kind") == "lycoris_lora"
                and flat.get("adapter.lycoris_algo") == "locon"
            )
        )
    ):
        out.pop("train.moss_qkv_format", None)
    if not (
        flat.get("train.model_family") in ("auto", "moss_image")
        and bool(flat.get("train.compile"))
        and bool(flat.get("train.gradient_checkpointing"))
    ):
        out.pop("train.moss_checkpoint_inside_compile", None)
    if flat.get('dataset.target_images') is not None:
        out.pop('dataset.num_repeats', None)
    return out


# ---------------------------------------------------------------- file I/O


def read_toml(path: str | Path) -> dict:
    # Match the trainer's parser, including nested arrays inside inline tables.
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


class _Inline(dict, toml.decoder.InlineTableDict):
    """Emit `batch_size = { 512 = 4, 1024 = 12 }` instead of a `[train.batch_size]` sub-table.

    Both parse identically, but the sub-table form has to be written *after* every scalar in the
    file and reads as a separate section, which is exactly wrong for what is one value.
    """



def _mark_inline(nested: dict) -> dict:
    """Leaf dicts (batch_size, mixed_weights) become inline tables; section dicts do not."""
    out = {}
    for key, value in nested.items():
        if isinstance(value, dict) and value and not any(
            isinstance(v, dict) for v in value.values()
        ) and key not in SECTIONS and key not in ("caption", "texture"):
            out[key] = _Inline({str(k): v for k, v in value.items()})
        elif isinstance(value, dict):
            out[key] = _mark_inline(value)
        else:
            out[key] = value
    return out


def dump_toml(flat: dict) -> str:
    nested = _mark_inline(nest(prune_defaults(flat)))
    if flat.get("adapter.kind") == "lycoris_lora" and "train.compile" in flat and flat["train.compile"] is None:
        nested.setdefault("train", {})["compile"] = False
    return toml.dumps(nested, encoder=toml.encoder.TomlPreserveInlineDictEncoder())


def write_toml(path: str | Path, flat: dict) -> None:
    Path(path).write_text(dump_toml(flat), encoding="utf-8")


def validate(flat: dict) -> tuple[bool, str]:
    """Run the trainer's own loader over a candidate config.

    Written to a temp file rather than constructing `Config` directly on purpose: several rules --
    `resolution` vs `resolutions` above all -- are enforced against the *raw TOML*, because only
    the file distinguishes "explicitly set" from "left at the default". Validating the object would
    silently skip them.
    """
    text = dump_toml(flat)
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(text)
        tmp = fh.name
    try:
        load_config(tmp)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        Path(tmp).unlink(missing_ok=True)


def summarize(flat: dict) -> str:
    """One-line description of what this config will do, for the status bar."""
    cfg_ok, _ = validate(flat)
    if not cfg_ok:
        return "config invalid"
    kind = flat.get("adapter.kind", "none")
    mode = "full finetune" if kind in (None, "none", "") else f"LoRA/{kind}"
    if kind == "lora" and float(flat.get("adapter.loraplus_lr_ratio") or 1.0) != 1.0:
        mode += f" (LoRA+ x{float(flat['adapter.loraplus_lr_ratio']):g})"
    tiers = flat.get("dataset.resolutions") or [flat.get("dataset.resolution", 1024)]
    quant = flat.get("quant.mode", "none")
    bits = [mode, f"{'/'.join(str(t) for t in tiers)}px"]

    if flat.get('resolution_schedule.enabled'):
        duration = ('explicit epochs per stage' if flat.get('resolution_schedule.unit') == 'stage_epochs'
                    else f"{flat.get('train.epochs')} epochs" if flat.get('resolution_schedule.unit') == 'epochs'
                    else f"{flat.get('train.max_steps')} successful updates")
        bits.append(f"{duration} / staged resolutions")
    elif flat.get("train.pack_resolutions") and flat.get("train.pack_mode") == "tokens":
        bits.append(f"packed tokens<={flat.get('train.pack_token_budget', 8576)}")
    # Whether a run crops is decided by the curriculum, which lives in a table on another tab --
    # so without this the status bar reads identically for a plain LoRA and a texture run.
    phases = flat.get("curriculum") or []
    if phases:
        n_tex = sum(1 for p in phases if p.get("mode") == "texture")
        bits.append(f"curriculum {len(phases)} phase(s)"
                    + (f", {n_tex} texture" if n_tex else ", no texture"))
        if not n_tex and flat.get("dataset.source") == "encode":
            bits.append("(source=encode with no texture phase: +16% step time for nothing)")
    elif flat.get("dataset.source") == "encode":
        bits.append("(source=encode with no curriculum: +16% step time for nothing)")

    if quant not in (None, "none"):
        if flat.get("quant.backend", "sdnq") == "convrot":
            bits.append("convrot frozen/int8 matmul")
        else:
            bits.append(f"sdnq {quant}/{flat.get('quant.weights_dtype', 'int8')}")
    if flat.get("train.compile"):
        bits.append(f"compile {flat['train.compile']}")
    return "  ".join(bits)


# ---------------------------------------------------------------- advisories


def _texture_phases(flat: dict) -> list[dict]:
    return [p for p in (flat.get("curriculum") or []) if p.get("mode") == "texture"]


def _restricted_share(flat: dict) -> tuple[float, float]:
    """(share of the run in a restricted t_range, run-average t^2 factor for the hf term).

    Under `rescale` -- the default -- a phase maps the full distribution onto [lo, hi], so
    t' = lo + (hi-lo)*t and with lo = 0 the hf term's t^2 factor is exactly (hi-lo)^2. That closed
    form is used rather than sampling, because this runs on every keystroke. Checked against a
    200k-sample draw: predicted 0.36x for [0, 0.6], measured 0.36x.
    """
    phases = flat.get("curriculum") or []
    if not phases:
        return 0.0, 1.0
    share, factor = 0.0, 0.0
    for i, p in enumerate(phases):
        end = phases[i + 1].get("at", 1.0) if i + 1 < len(phases) else 1.0
        span = max(0.0, end - p.get("at", 0.0))
        lo, hi = p.get("t_range", (0.0, 1.0))
        width = hi - lo
        # lo^2 + 2*lo*(hi-lo)*E[t] + (hi-lo)^2*E[t^2], with the run's own E[t]/E[t^2].
        # E[t]=0.676 and E[t^2]=0.528 for uniform+shift 3, this trainer's measured default.
        f = lo * lo + 2 * lo * width * 0.676 + width * width * 0.528
        factor += span * (f / 0.528)
        if width < 1.0 or lo > 0.0:
            share += span
    return share, factor


def advisories(flat: dict, num_processes: int = 1) -> list[tuple[str, str]]:
    """(level, message) for combinations that load fine and then train badly.

    Distinct from `validate`, which surfaces what the trainer would REJECT. Everything here is
    legal and runnable; it is just known -- from a measurement, not a hunch -- to be a bad idea, or
    to mean something different from what it looks like. Each message carries the number that
    justifies it, because an unexplained warning is one people learn to click past.

    `num_processes` is a GUI-only input: the config cannot express how many GPUs are ticked, and
    the two worst traps here are both about that.
    """
    out: list[tuple[str, str]] = []
    tex = _texture_phases(flat)
    n = max(1, int(num_processes))

    if (flat.get("optimizer.kind") == "adafactor"
            and flat.get("optimizer.norm_mode") in (None, "relative")
            and flat.get("adapter.kind", "none") != "none"):
        out.append(("warn", "SDNQ Adafactor relative normalization can nearly stall zero-initialized "
                    "LoRA projections at small learning rates. Select rms_clip and tune the learning rate; "
                    "gradient clipping is a separate setting."))

    if n > 1 and flat.get("optimizer.gradient_release"):
        out.append(("error", "Optimi gradient release requires one GPU; DDP updates can precede gradient synchronization."))
    if n > 1:
        if tex and not flat.get("train.allow_multi_gpu_texture"):
            out.append(("error",
                        (f"{n} GPUs with a texture curriculum is REFUSED. Two runs here degraded "
                        f"anatomy progressively against single-GPU at matched steps -- so it is "
                        f"not undertraining, and more steps do not fix it. The cause is not "
                        f"identified; ranks seeding identically was found and fixed, but is not "
                        f"proven to be it. Use one GPU, or set train.allow_multi_gpu_texture to "
                        f"re-test the fix deliberately.")))
        elif tex:
            out.append(("warn",
                        (f"{n} GPUs with a texture curriculum, override ON. This configuration "
                        f"produced progressively worsening anatomy twice. You are re-testing an "
                        f"unproven fix -- compare against a single-GPU run at matched steps.")))
        if flat.get('resolution_schedule.enabled'):
            message = (f'{n} GPUs multiply presentations by {n} at a fixed staged step budget. '
                       if flat.get('resolution_schedule.unit', 'steps') == 'steps' else
                       f'{n} GPUs keep requested stage epochs fixed; update counts and final-batch padding change. '
                       'Increasing epochs increases dataset exposure. ')
            out.append(('note', message + 'Calculate dataset exposure before starting.'))
        else:
            out.append(("note",
                        (f"{n} GPUs means {n}x FEWER optimizer steps than the same config on one, at "
                        f"{n}x the effective batch. Multiply epochs by {n} to match a single-GPU run.")))

    if tex:
        bs = flat.get("train.batch_size", 1)
        if isinstance(bs, dict):
            bs = max(bs.values()) if bs else 1
        if bs and bs > 1:
            out.append(("note",
                        (f"batch_size {bs} in texture mode buys no throughput: measured 0.89-0.93 "
                        f"images/s flat from batch 1 to 8, while peak VRAM went 8.6 -> 14.6GB. A "
                        f"1024px canvas already saturates the GPU. It also shrinks the set of "
                        f"canvases every image in the batch can hold, so padding returns.")))
        tiers = flat.get("dataset.resolutions") or []
        if len(tiers) > 1:
            out.append(("note",
                        (f"{len(tiers)} resolution tiers with a texture curriculum: tiers set the "
                        f"FULLRES bucket, but a texture crop overrides shape with the canvas. So "
                        f"this costs {len(tiers)}x the epoch length and only benefits the fullres "
                        f"phases. Vary `texture.canvases` instead -- fit-aware selection makes a "
                        f"mixed-scale list adaptive per image.")))
        if flat.get("dataset.texture.oversize") == "cover":
            out.append(("warn",
                        ("oversize='cover' upscales without a ceiling. It lost its A/B here: it "
                        "fired on 57% of the texture set draws at a mean 1.25x (max 1.85x) and softened "
                        "faces. `cover_max_scale` is the capped version and is already on.")))

    hf = flat.get("flow.hf_scale") or 0.0
    if hf > 0.0:
        share, factor = _restricted_share(flat)
        if flat.get("flow.phase_mapping", "rescale") == "rescale" and factor < 0.9:
            out.append(("note",
                        (f"hf loss carries a t^2 factor and {share:.0%} of this run sits in a "
                        f"restricted t_range, so the term lands at {factor:.2f}x the strength the "
                        f"same hf_scale gives on a full-range run. hf_scale "
                        f"{hf / max(factor, 0.01):.2f} would match.")))
        if tex and flat.get("dataset.texture.pad_mode") == "black" \
                and not flat.get("dataset.texture.fit_aware", True):
            out.append(("warn",
                        ("hf loss with black padding and fit-aware selection off: the image/black "
                        "boundary is a maximal Laplacian edge, so detail weighting concentrates on "
                        "exactly the artifact you do not want learned. Padding fires on 56% of "
                        "draws in this configuration.")))

    if not flat.get("resolution_schedule.enabled") and flat.get("resolution_schedule.stages"):
        out.append(("note",
                    "resolution_schedule.stages is inactive while staged resolution training is off "
                    "and will be omitted when saving."))

    if (
        flat.get("train.pack_resolutions")
        and flat.get("train.pack_mode") == "tokens"
        and not flat.get("resolution_schedule.enabled")
    ):
        tiers = flat.get("dataset.resolutions") or [flat.get("dataset.resolution", 1024)]
        valid_tiers = [t for t in tiers if isinstance(t, int) and t > 0]
        budget = flat.get("train.pack_token_budget", 8576)
        if valid_tiers and isinstance(budget, int):
            max_img_tokens = max(valid_tiers) ** 2 // 256
            if budget < max_img_tokens:
                out.append(("warn",
                            f"pack_token_budget ({budget}) is smaller than the {max(valid_tiers)}px "
                            f"tier's image token count ({max_img_tokens}). Images in that tier will run "
                            f"as un-packed singletons."))

    if flat.get("flow.use_ot"):
        bs = flat.get("train.batch_size", 1)
        if (max(bs.values()) if isinstance(bs, dict) and bs else bs) == 1:
            out.append(("note",
                        "use_ot is a no-op at batch size 1 -- there is nothing to permute."))

    return out


def missing_ui_keys(ui_keys) -> tuple[list[str], list[str]]:
    """(config keys with no widget, widget keys that are not config keys).

    The gate asserts both are empty, which is what stops a new dataclass field from being
    unreachable from the GUI -- the exact failure mode that made `keep_last_n` and `torch.compile`
    dead keys in the TOML path.
    """
    known = set(schema()) | set(ARRAY_KEYS)
    ui = set(ui_keys)
    return sorted(known - ui - CLI_ONLY_KEYS), sorted(ui - known)
