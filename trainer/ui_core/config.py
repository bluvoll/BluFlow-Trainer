"""Canonical field descriptions and edit transitions for both desktop interfaces."""

import copy
import sys
from dataclasses import asdict
from ..gui import bridge
from ..training.optimizer_specs import OPTIMIZERS, optimizer_defaults
from ..training.quant import QUANT_PRESETS
from ..training.flow import FlowConfig
from ..training.resolution_schedule import ResolutionSchedule
from .schema import SPEC, LAYOUT
from .rules import _RULES
from .timestep import distribution, SAMPLER_KEYS
from .presets import TRAINTRAIN_CURRICULUM


def normalize(flat, previous=None):
    value = copy.deepcopy(flat)
    if previous is not None:
        if value.get("resolution_schedule.enabled") and not previous.get(
            "resolution_schedule.enabled"
        ):
            tiers = value.get("dataset.resolutions") or [512, 768, 1024]
            if not value.get("resolution_schedule.stages"):
                shares = (
                    [0.2, 0.2, 0.6]
                    if len(tiers) == 3
                    else [round(100 / len(tiers), 4) / 100] * len(tiers)
                )
                shares[-1] = 1 - sum(shares[:-1])
                value["resolution_schedule.stages"] = [
                    dict(resolution=t, share=w, lr_mul={512: 0.8, 768: 0.9}.get(t, 1.0))
                    for t, w in zip(tiers, shares)
                ]
            value["dataset.resolutions"] = tiers
            if not isinstance(value.get("train.batch_size"), dict):
                value["train.batch_size"] = {
                    t: value.get("train.batch_size", 1) for t in tiers
                }
            value["train.pack_resolutions"] = True
            if value.get("train.attention_backend") == "sdpa":
                value["train.attention_backend"] = (
                    "flash_attn_2" if sys.platform == "win32" else "torch_varlen"
                )
            if not value.get("train.max_steps"):
                value["train.max_steps"] = 560
        kind = value.get("optimizer.kind")
        if kind != previous.get("optimizer.kind") and kind in OPTIMIZERS:
            value.update(
                {f"optimizer.{k}": v for k, v in optimizer_defaults(kind).items()}
            )
        selection = (value.get("adapter.kind"), value.get("adapter.lycoris_algo"))
        if (
            selection
            != (previous.get("adapter.kind"), previous.get("adapter.lycoris_algo"))
            and selection[0] == "lycoris_lora"
        ):
            value.update(
                {
                    "train.compile": "default",
                    "adapter.lycoris_bypass": selection[1] != "dora",
                    "adapter.lycoris_wd_on_output": True,
                }
            )
    value.update(
        {
            f"quant.{k}": v
            for k, v in QUANT_PRESETS.get(value.get("quant.preset"), {}).items()
        }
    )
    if value.get("flow.timestep_sample_method") == "moss":
        value.update(
            {"flow.sigmoid_scale": 1.0, "flow.shift": 1.0, "flow.flux_shift": False}
        )
    if value.get("dataset.subsets"):
        value["dataset.path"] = None
    for key in ("train.batch_size",):
        if isinstance(value.get(key), dict):
            value[key] = {int(k): v for k, v in value[key].items()}
    return value


def descriptors():
    fields = {}
    for key, spec in SPEC.items():
        editor = spec.make().describe()
        if editor["kind"] == "OptimizerEditor":
            editor["args"] = [
                [f"{s.name} ({s.family})" for s in OPTIMIZERS.values()],
                list(OPTIMIZERS),
            ]
        fields[key] = dict(
            key=key,
            label=spec.label,
            tooltip=spec.tooltip,
            inline=spec.inline_label,
            **editor,
        )
    pages = []
    experiments = []
    for title, groups in LAYOUT:
        regular = []
        for group, keys in groups:
            if group.startswith("Experimental "):
                experiments.append([group, keys])
                continue
            dual = [k for k in keys if k.startswith("flow.dual_timestep")]
            if dual:
                experiments.append(["Dual timestep noising", dual])
            regular.append([group, [k for k in keys if k not in dual]])
        pages.append([title, regular])
    pages.append(["Experimental", experiments])
    return dict(fields=fields, pages=pages, curriculumPreset=TRAINTRAIN_CURRICULUM)


def evaluate(flat, processes=1, show_experimental=False, latent_h=64, latent_w=64):
    enabled = {k: True for k in SPEC}
    locked = {f"quant.{k}" for k in QUANT_PRESETS.get(flat.get("quant.preset"), {})}
    for key, rule in _RULES.items():
        try:
            enabled[key] = bool(rule(flat)) and key not in locked
        except (ValueError, TypeError):
            enabled[key] = True
    if flat.get("dataset.subsets"):
        enabled["dataset.path"] = enabled["dataset.num_repeats"] = False
    try:
        ok, error = bridge.validate(flat)
        notes = bridge.advisories(flat, processes)
        summary = bridge.summarize(flat) if ok else "Review configuration"
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        ok, error, notes, summary = False, str(exc), [], "Review configuration"
    preview = None
    try:
        cfg = FlowConfig(
            **{
                k.removeprefix("flow."): v
                for k, v in flat.items()
                if k.startswith("flow.") and k.removeprefix("flow.") in SAMPLER_KEYS
            }
        )
        preview = asdict(distribution(cfg, latent_h, latent_w))
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        pass
    stages = None
    if flat.get("resolution_schedule.enabled"):
        try:
            budgets = ResolutionSchedule(
                enabled=True, stages=flat["resolution_schedule.stages"]
            ).budgets(flat.get("train.max_steps"))
            stages = [
                dict(
                    **stage,
                    steps=budget,
                    effective_batch=(
                        flat["train.batch_size"][stage["resolution"]]
                        * processes
                        * flat.get("train.gradient_accumulation_steps", 1)
                    ),
                )
                for stage, budget in zip(flat["resolution_schedule.stages"], budgets)
            ]
        except (ValueError, TypeError, KeyError, ZeroDivisionError):
            pass
    experiments = show_experimental or any(
        flat.get(k)
        for k in (
            "self_flow.enabled",
            "flow.dual_timestep",
            "rti.enabled",
            "perceptual.enabled",
            "adversarial.enabled",
            "train.moss_checkpoint_inside_compile",
            "block_lr.enabled",
        )
    )
    windows_repair = sys.platform == "win32" and (
        flat.get("train.attention_backend") == "torch_varlen"
        or (
            flat.get("train.pack_resolutions")
            and flat.get("train.attention_backend") != "flash_attn_2"
        )
    )
    return dict(
        valid=ok,
        error=error,
        advisories=notes,
        summary=summary,
        enabled=enabled,
        experimental=bool(experiments),
        distribution=preview,
        stages=stages,
        windowsRepair=bool(windows_repair),
    )
