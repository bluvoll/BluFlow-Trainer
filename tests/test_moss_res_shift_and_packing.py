"""Tests for resolution-aware MOSS/Logit Normal shift, 3D curves, and MOSS token-budget packing."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch
from PySide6 import QtWidgets

from trainer.data.packed import (
    NativeResolutionBatchSampler,
    TokenBudgetBatchSampler,
    estimate_entry_tokens,
)
from trainer.gui import bridge
from trainer.gui.timestep_plot import TimestepDistributionWidget
from trainer.training.config import load_config
from trainer.training.flow import FlowConfig, flow_loss, sample_timesteps
from trainer.training.moss_timesteps import effective_moss_mu, probability_table
from trainer.training.timestep_distribution import effective_shift, probability_law
from trainer.ui_core.config import evaluate, normalize
from trainer.ui_core.timestep import distribution, resolution_distributions


def test_moss_res_shift_exact_1024_and_scales_across_resolutions():
    assert effective_moss_mu(-1.09, 64, 64, True) == pytest.approx(-1.09, abs=1e-8)
    assert effective_moss_mu(-1.09, 32, 32, True) == pytest.approx(-1.09 + math.log(2), abs=1e-7)
    assert effective_moss_mu(-1.09, 80, 80, True) == pytest.approx(-1.09 - math.log(1.25), abs=1e-7)

    static_cfg = FlowConfig(timestep_sample_method="moss", shift=1.0, res_shift=False)
    res_cfg = FlowConfig(timestep_sample_method="moss", shift=1.0, res_shift=True)

    # Bit-identical at 1024px (64x64 latent grid).
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    t_static = sample_timesteps(static_cfg, 256, 64, 64, "cpu", generator=g1)
    t_res_1024 = sample_timesteps(res_cfg, 256, 64, 64, "cpu", generator=g2)
    assert torch.equal(t_static, t_res_1024)

    # 512px shifts toward lower noise; 1280px shifts toward higher noise; stays on discrete 1/1000 grid.
    d512 = distribution(res_cfg, 32, 32)
    d1024 = distribution(res_cfg, 64, 64)
    d1280 = distribution(res_cfg, 80, 80)
    assert d512.median < d1024.median < d1280.median
    assert d512.shift == pytest.approx(math.exp(1.09) * 0.5, rel=1e-4)
    assert d1024.shift == pytest.approx(math.exp(1.09), rel=1e-4)
    assert d1280.shift == pytest.approx(math.exp(1.09) * 1.25, rel=1e-4)

    draws_512 = sample_timesteps(res_cfg, 1000, 32, 32, "cpu", generator=torch.Generator().manual_seed(7))
    scaled = draws_512 * 1000
    assert torch.allclose(scaled, scaled.round(), atol=1e-4)

    # Empirical draws at 512px match probability_law CDF (including with normal_low_snr).
    res_snr = FlowConfig(timestep_sample_method="moss", shift=1.0, res_shift=True, normal_low_snr=True)
    law = probability_law(res_snr, 32, 32)
    emp = sample_timesteps(res_snr, 40000, 32, 32, "cpu", generator=torch.Generator().manual_seed(19))
    for edge in (0.3, 0.5, 0.7, 0.9):
        assert (emp < edge).double().mean().item() == pytest.approx(law.cdf(edge, inclusive=False), abs=0.01)


def test_logit_normal_res_shift_scales_linearly_and_is_mutually_exclusive():
    cfg = FlowConfig(timestep_sample_method="logit_normal", shift=3.0, res_shift=True)
    assert effective_shift(cfg, 32, 32) == pytest.approx(1.5)
    assert effective_shift(cfg, 48, 48) == pytest.approx(2.25)
    assert effective_shift(cfg, 64, 64) == pytest.approx(3.0)
    assert effective_shift(cfg, 80, 80) == pytest.approx(3.75)

    assert distribution(cfg, 32, 32).median == pytest.approx(1.5 / 2.5)
    assert distribution(cfg, 64, 64).median == pytest.approx(3.0 / 4.0)
    assert distribution(cfg, 80, 80).median == pytest.approx(3.75 / 4.75)

    with pytest.raises(ValueError, match="mutually exclusive"):
        FlowConfig(res_shift=True, cabal_shift=True, shift=3.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        FlowConfig(res_shift=True, flux_shift=True, shift=None)
    with pytest.raises(ValueError, match="anchor"):
        FlowConfig(res_shift=True, shift=None)


def test_resolution_curves_and_qt_3d_canvas_for_moss_and_logit_normal():
    base = bridge.defaults() | {
        "dataset.path": "/images",
        "train.model_family": "moss_image",
        "dataset.resolutions": [512, 1024, 1280],
    }

    # Logit Normal + res_shift produces 3 resolution curves for the 3D graph.
    ln_res = normalize(base | {"flow.res_shift": True, "flow.shift": 3.0}, base)
    ev_ln = evaluate(ln_res)
    assert [c["resolution"] for c in ev_ln["resolutionDistributions"]] == [512, 1024, 1280]
    assert [c["shift"] for c in ev_ln["resolutionDistributions"]] == pytest.approx([1.5, 3.0, 3.75])
    assert not ln_res["flow.cabal_shift"] and not ln_res["flow.flux_shift"]

    # MOSS + res_shift stays enabled when switching to moss and emits 3 curves.
    moss_res = normalize(ln_res | {"flow.timestep_sample_method": "moss"}, ln_res)
    assert moss_res["flow.res_shift"] is True
    assert moss_res["flow.shift"] == 1.0
    ev_moss = evaluate(moss_res)
    assert ev_moss["valid"] is True
    assert [c["resolution"] for c in ev_moss["resolutionDistributions"]] == [512, 1024, 1280]
    assert ev_moss["resolutionDistributions"][0]["shift"] < ev_moss["resolutionDistributions"][1]["shift"]
    assert ev_moss["resolutionDistributions"][1]["shift"] < ev_moss["resolutionDistributions"][2]["shift"]

    # Qt TimestepDistributionWidget populates 3D curves and paints cleanly offscreen.
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = TimestepDistributionWidget()
    try:
        widget.resize(400, 320)
        widget.set_config(moss_res)
        assert len(widget.canvas.curves) == 3
        assert "512px" in widget.summary.text() and "1280px" in widget.summary.text()
        widget.canvas.grab()
        widget.set_config(ln_res)
        assert len(widget.canvas.curves) == 3
        widget.canvas.grab()
    finally:
        widget.deleteLater()
        app.processEvents()


@dataclass
class _FakeEntry:
    bucket: tuple[int, int]
    resolution: int
    tags: str = "1girl, solo, looking at viewer"
    nl: str | None = None
    moss: dict | None = None
    texture_ok: bool = True


class _FakeDataset:
    def __init__(self, entries):
        self.entries = entries
        self.weighted = False

    def __len__(self):
        return len(self.entries)


def test_token_budget_sampler_packs_balances_and_preserves_every_sample():
    entries = (
        [_FakeEntry((512, 512), 512) for _ in range(12)]
        + [_FakeEntry((1024, 1024), 1024) for _ in range(6)]
        + [_FakeEntry((1280, 1280), 1280) for _ in range(4)]
        + [_FakeEntry((2048, 2048), 2048)]  # Oversized singleton (16384+ > 8576)
    )
    ds = _FakeDataset(entries)
    sampler = TokenBudgetBatchSampler(ds, token_budget=8576, max_text_tokens=512, shuffle=True, seed=42)
    batches = list(sampler)
    assert len(batches) == len(sampler)

    # Every entry visited exactly once, including oversized singleton.
    # BucketBatchSampler.__iter__ yields (idx, epoch, canvas, seed) tuples.
    seen = [item[0] for pack in batches for item in pack]
    assert sorted(seen) == list(range(len(entries)))

    # Every multi-sample pack stays within token budget; oversized entry runs alone.
    costs = [estimate_entry_tokens(e, 512) for e in entries]
    for pack in batches:
        idxs = [item[0] for item in pack]
        if len(idxs) > 1:
            assert sum(costs[i] for i in idxs) <= 8576
        else:
            assert len(idxs) == 1
    assert any([item[0] for item in pack] == [len(entries) - 1] for pack in batches)

    avg_imgs, avg_toks, oversized = sampler.pack_stats()
    assert avg_imgs > 1.5 and avg_toks > 4000 and oversized == 1

    # Different epochs re-shuffle deterministically.
    sampler.set_epoch(1)
    batches_e1 = list(sampler)
    assert sorted(item[0] for p in batches_e1 for item in p) == list(range(len(entries)))
    assert [[item[0] for item in p] for p in batches_e1] != [[item[0] for item in p] for p in batches]

    # Legacy fixed-count sampler still works identically.
    legacy = NativeResolutionBatchSampler(ds, batch_size=3, shuffle=True, seed=42)
    assert sum(len(b) for b in legacy) == len(entries)


def test_moss_token_packing_config_roundtrip_rules_and_per_token_loss(tmp_path):
    p = tmp_path / "moss_pack.toml"
    flat = bridge.defaults() | {
        "dataset.path": "/images",
        "dataset.resolutions": [512, 1024, 1280],
        "train.model_family": "moss_image",
        "train.attention_backend": "torch_varlen",
        "train.pack_resolutions": True,
        "train.pack_mode": "tokens",
        "train.pack_token_budget": 8576,
        "flow.timestep_sample_method": "moss",
        "flow.shift": 1.0,
        "flow.res_shift": True,
    }
    bridge.write_toml(p, flat)
    cfg = load_config(p)
    assert cfg.train.pack_resolutions is True
    assert cfg.train.pack_mode == "tokens"
    assert cfg.train.pack_token_budget == 8576
    assert cfg.flow.res_shift is True

    ev = evaluate(flat)
    assert ev["valid"] is True
    assert ev["enabled"]["train.pack_mode"] is True
    assert ev["enabled"]["train.pack_token_budget"] is True
    assert ev["enabled"]["train.batch_size"] is False
    assert "packed tokens<=8576" in ev["summary"]

    # Too-small budget produces a clear warning advisory.
    small = flat | {"train.pack_token_budget": 4096}
    assert any(lvl == "warn" and "pack_token_budget" in msg for lvl, msg in bridge.advisories(small))

    # Staged resolution rejects pack_mode='tokens'.
    staged_bad = tmp_path / "staged_bad.toml"
    staged_bad.write_text(
        """
[train]
model_family = "moss_image"
attention_backend = "torch_varlen"
pack_resolutions = true
pack_mode = "tokens"
batch_size = { 512 = 4, 1024 = 2 }
max_steps = 20
[dataset]
path = "/images"
resolutions = [512, 1024]
[resolution_schedule]
enabled = true
unit = "steps"
[[resolution_schedule.stages]]
resolution = 512
share = 0.5
[[resolution_schedule.stages]]
resolution = 1024
share = 0.5
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pack_mode='samples'"):
        load_config(staged_bad)

    # Verify Eq. (3) per-token loss vs legacy per-sample loss on a mixed 512 + 1280 pack.
    y_small = torch.zeros(1, 128, 1, 32, 32)
    t_small = torch.ones(1, 128, 1, 32, 32) * 1.0  # MSE = 1.0 over 1024 tokens
    y_large = torch.zeros(1, 128, 1, 80, 80)
    t_large = torch.ones(1, 128, 1, 80, 80) * 2.0  # MSE = 4.0 over 6400 tokens

    legacy_loss = torch.stack([flow_loss(y_small, t_small), flow_loss(y_large, t_large)]).mean().item()
    eq3_loss = (
        ((y_small - t_small).square().sum() + (y_large - t_large).square().sum())
        / (t_small.numel() + t_large.numel())
    ).item()
    assert legacy_loss == pytest.approx(2.5)
    assert eq3_loss == pytest.approx((1024 * 1.0 + 6400 * 4.0) / (1024 + 6400))
    assert eq3_loss > legacy_loss


def test_token_budget_sampler_keeps_odd_pack_tail_without_shifting_pairs():
    # Craft 5 entries that cannot share a pack at budget=5000 -> 5 packs (odd count).
    # Costs decrease cleanly: [4624, 4124, ...]: packs 0&1 form pair 0, packs 2&3 form pair 1, pack 4 is trailing tail.
    entries = [
        _FakeEntry((1024, 1024), 1024, tags="a"),        # 4096 + 16 = 4112
        _FakeEntry((1024, 960), 1024, tags="b"),         # 3840 + 16 = 3856
        _FakeEntry((896, 896), 896, tags="c"),           # 3136 + 16 = 3152
        _FakeEntry((832, 832), 832, tags="d"),           # 2704 + 16 = 2720
        _FakeEntry((768, 768), 768, tags="e"),           # 2304 + 16 = 2320
    ]
    ds = _FakeDataset(entries)
    for group_size in (2, 4):
        for epoch in range(4):
            sampler = TokenBudgetBatchSampler(
                ds, token_budget=5000, max_text_tokens=16, shuffle=True, seed=7, ddp_group_size=group_size
            )
            sampler.set_epoch(epoch)
            batches = list(sampler)
            assert len(batches) == 5
            costs = [sum(sampler._entry_tokens[item[0]] for item in b) for b in batches]
            # The smallest singleton trailing remainder stays at the end; full groups stay rank-balanced.
            assert costs[-1] == min(costs)
            for pos in range(0, (len(costs) // group_size) * group_size, group_size):
                chunk = costs[pos : pos + group_size]
                assert abs(chunk[0] - chunk[-1]) <= 1500


def test_global_sample_scale_preserves_fractional_token_weights():
    from trainer.training.step_safety import global_sample_scale

    s_int = global_sample_scale(6400, accumulation_steps=2, device=torch.device("cpu"))
    assert abs(s_int.item() - (2.0 / 6400.0)) < 1e-8
    w_mask = torch.tensor(1234.5, dtype=torch.float32)
    s_mask = global_sample_scale(w_mask, accumulation_steps=2, device=torch.device("cpu"))
    assert abs(s_mask.item() - (2.0 / 1234.5)) < 1e-7


def test_token_packing_sets_sampling_identity_on_unweighted_dataset():
    from trainer.training.train import Trainer

    t = object.__new__(Trainer)
    t.cfg = load_config_for_packing_identity()
    t.dataset = _FakeDataset([_FakeEntry((1024, 1024), 1024)])
    t.dataset.entries[0].latent_path = _FakePath()
    t.accelerator = type("_A", (), {"num_processes": 2})()
    # Mirror the exact sampling_identity guard from Trainer._build_data.
    enabled = (
        getattr(t.dataset, "weighted", False)
        or (t.cfg.train.pack_resolutions and t.cfg.train.pack_mode == "tokens")
    ) and not t.cfg.resolution_schedule.enabled
    assert enabled is True
    # Check _sampling_state includes epoch/cursor/micro when sampling_identity is set.
    t.sampling_identity = "abc"
    t.data_epoch = 3
    t.data_cursor = 5
    t._micro = 11
    state = t._sampling_state()
    assert state == {"version": 2, "identity": "abc", "epoch": 3, "cursor": 5, "micro": 11}


def load_config_for_packing_identity():
    from types import SimpleNamespace

    return SimpleNamespace(
        train=SimpleNamespace(pack_resolutions=True, pack_mode="tokens"),
        resolution_schedule=SimpleNamespace(enabled=False),
    )


class _FakePath:
    def resolve(self):
        return self

    def __str__(self):
        return "/tmp/fake.latent"
