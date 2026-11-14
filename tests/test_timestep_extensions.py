"""Cabal reference numerics, distribution laws, and shared desktop controls."""

import math
import os
from dataclasses import asdict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch

from trainer.training.flow import FlowConfig, sample_timesteps, apply_static_shift
from trainer.training.timestep_distribution import cabal_shift, probability_law
from trainer.ui_core.timestep import distribution, resolution_distributions


@pytest.mark.parametrize(
    "size,raw,anchored",
    [
        (256, 0.33, 1.205),
        (512, 1.0, 2.75),
        (768, 1.625, 4.25),
        (1024, 2.5, 6.0),
        (1536, 5.0, 10.25),
        (2048, 8.5, 15.5),
    ],
)
def test_cabal_reference_curve(size, raw, anchored):
    # Golden values checked against calculate_dynamic_timestep_shift in the SDXL trainer.
    assert cabal_shift(size / 16, size / 16, 2.5) == pytest.approx(raw)
    assert cabal_shift(size / 16, size / 16, 6) == pytest.approx(anchored)


def test_cabal_rectangles_offset_and_curve_bounds():
    assert cabal_shift(32, 128, 6) == 6
    assert cabal_shift(64, 64, 6, 0.25) == 6.25
    assert cabal_shift(32, 32, 6, 0.25) == 2.875
    assert cabal_shift(256, 256, 6) == 22.5
    with pytest.raises(ValueError, match="positive"):
        cabal_shift(16, 16, 1)


@pytest.mark.parametrize("method", ["uniform", "logit_normal", "moss"])
def test_default_draws_and_rng_unchanged(method):
    from trainer.training.moss_timesteps import inverse_cdf

    cfg = FlowConfig(timestep_sample_method=method, shift=1 if method == "moss" else 6)
    reference = torch.Generator().manual_seed(72)
    if method == "logit_normal":
        expected = torch.sigmoid(
            torch.randn(512, generator=reference) * cfg.sigmoid_scale
        )
    else:
        expected = torch.rand(512, generator=reference)
        if method == "moss":
            expected = inverse_cdf(expected, cfg)
    expected = apply_static_shift(expected, cfg.shift)
    generator = torch.Generator().manual_seed(72)
    result = sample_timesteps(cfg, 512, 64, 64, "cpu", generator=generator)
    assert torch.equal(result, expected)
    assert torch.equal(generator.get_state(), reference.get_state())


def test_logit_mean_std_and_quantiles():
    cfg = FlowConfig(logit_mean=-0.2, sigmoid_scale=1.5, shift=1)
    expected = torch.sigmoid(
        torch.randn(100, generator=torch.Generator().manual_seed(5)) * 1.5 - 0.2
    )
    actual = sample_timesteps(
        cfg, 100, 64, 64, "cpu", generator=torch.Generator().manual_seed(5)
    )
    assert torch.equal(actual, expected)
    assert distribution(cfg).median == pytest.approx(1 / (1 + math.exp(0.2)))
    assert sample_timesteps(
        cfg, 1, 64, 64, "cpu", quantile=0.5
    ).item() == pytest.approx(distribution(cfg).median)


@pytest.mark.parametrize("method", ["uniform", "logit_normal", "moss"])
@pytest.mark.parametrize("mapping", ["rescale", "truncate"])
@pytest.mark.parametrize("bounds", [None, (0.2, 0.6), (0.4, 0.99)])
def test_low_snr_empirical_law_ranges_and_rng(method, mapping, bounds):
    cfg = FlowConfig(
        timestep_sample_method=method,
        shift=1 if method == "moss" else 0.5,
        sigmoid_scale=1,
        normal_low_snr=True,
        phase_mapping=mapping,
        moss_mu=0,
        moss_std=0.5,
    )
    law = probability_law(cfg, 64, 64, bounds)
    generator = torch.Generator().manual_seed(91)
    start = generator.get_state().clone()
    global_state = torch.get_rng_state().clone()
    values = sample_timesteps(
        cfg, 60000, 64, 64, "cpu", generator=generator, t_range=bounds
    )
    assert torch.equal(global_state, torch.get_rng_state())
    generator.set_state(start)
    assert torch.equal(
        values,
        sample_timesteps(
            cfg, 60000, 64, 64, "cpu", generator=generator, t_range=bounds
        ),
    )
    lo, hi = bounds or (0, 1)
    assert values.min() >= lo - 1e-7 and values.max() <= hi + 1e-7
    for index in range(1, 20):
        edge = index / 20
        # Use >=-side bin convention; strict CDF handles MOSS atoms on edges.
        empirical = (
            (values < torch.tensor(edge, dtype=values.dtype)).double().mean().item()
        )
        assert empirical == pytest.approx(law.cdf(edge, inclusive=False), abs=0.009)
    if method == "moss":
        base_points = torch.tensor(law.points, dtype=values.dtype)
        positions = torch.searchsorted(base_points, values).clamp_max(
            len(base_points) - 1
        )
        distance = torch.minimum(
            (base_points[positions] - values).abs(),
            (base_points[(positions - 1).clamp_min(0)] - values).abs(),
        )
        assert distance.max() <= torch.finfo(values.dtype).eps
    for q in (0.1, 0.5, 0.9):
        probe = sample_timesteps(
            cfg, 1, 64, 64, "cpu", quantile=q, t_range=bounds
        ).item()
        if law.move_probability:
            assert probe == pytest.approx(law.quantile(q), abs=1e-7)
        else:
            original = FlowConfig(**(asdict(cfg) | {"normal_low_snr": False}))
            assert (
                probe
                == sample_timesteps(
                    original, 1, 64, 64, "cpu", quantile=q, t_range=bounds
                ).item()
            )


def test_low_snr_fills_tail_without_making_uniform_or_pure_noise_change():
    cfg = FlowConfig(shift=1, normal_low_snr=True)
    p = distribution(cfg, bins=1000)
    assert sum(p.probabilities) == pytest.approx(1)
    assert min(p.probabilities) >= 0
    assert p.probabilities[950:] == pytest.approx([0.001] * 50, abs=1e-12)
    original = distribution(FlowConfig(shift=1), bins=1000)
    assert sum(p.probabilities[:950]) < sum(original.probabilities[:950])
    for method, options in [("uniform", {}), ("moss", {"moss_p_full": 1})]:
        cfg = FlowConfig(
            timestep_sample_method=method, shift=1, normal_low_snr=True, **options
        )
        before = torch.Generator().manual_seed(9)
        after = torch.Generator().manual_seed(9)
        old = FlowConfig(**(asdict(cfg) | {"normal_low_snr": False}))
        assert torch.equal(
            sample_timesteps(cfg, 100, 64, 64, "cpu", generator=before),
            sample_timesteps(old, 100, 64, 64, "cpu", generator=after),
        )
        assert torch.equal(before.get_state(), after.get_state())


@pytest.mark.parametrize(
    "options",
    [
        {"logit_mean": float("nan")},
        {"sigmoid_scale": float("inf")},
        {"sigmoid_scale": 0},
        {"cabal_shift": True, "flux_shift": True, "shift": None},
        {"cabal_shift": True, "shift": None},
        {"cabal_shift_offset": float("nan")},
        {"timestep_sample_method": "moss", "shift": 1, "cabal_shift": True},
    ],
)
def test_invalid_sampler_settings(options):
    with pytest.raises(ValueError):
        FlowConfig(**options)


def test_resolution_curves_and_edit_transitions():
    from trainer.gui import bridge
    from trainer.ui_core.config import normalize, evaluate

    flat = bridge.defaults() | {
        "dataset.path": "/images",
        "dataset.resolutions": [1024, 512, 768],
    }
    cabal = normalize(flat | {"flow.cabal_shift": True}, flat)
    preview = evaluate(cabal)
    assert [c["resolution"] for c in preview["resolutionDistributions"]] == [
        512,
        768,
        1024,
    ]
    assert [c["shift"] for c in preview["resolutionDistributions"]] == [2.75, 4.25, 6]
    assert all(
        len(c["probabilities"]) == 100 for c in preview["resolutionDistributions"]
    )
    flux = normalize(cabal | {"flow.flux_shift": True}, cabal)
    assert not flux["flow.cabal_shift"] and flux["flow.shift"] is None
    back = normalize(flux | {"flow.cabal_shift": True}, flux)
    assert not back["flow.flux_shift"] and back["flow.shift"] == 6
    moss = normalize(
        back | {"flow.timestep_sample_method": "moss", "flow.normal_low_snr": True},
        back,
    )
    assert not moss["flow.cabal_shift"] and not moss["flow.flux_shift"]
    assert moss["flow.normal_low_snr"] and moss["flow.shift"] == 1
    assert not evaluate(moss)["enabled"]["flow.logit_mean"]
    staged = flat | {
        "resolution_schedule.enabled": True,
        "resolution_schedule.stages": [{"resolution": 768}, {"resolution": 512}],
    }
    assert [
        c["resolution"]
        for c in resolution_distributions(FlowConfig(cabal_shift=True), staged)
    ] == [512, 768]


def test_roundtrip_and_reject_nonpositive_configured_resolution(tmp_path):
    from trainer.gui import bridge
    from trainer.training.config import load_config

    flat = bridge.defaults() | {
        "dataset.path": "/images",
        "train.model_family": "moss_image",
        "flow.logit_mean": -0.2,
        "flow.sigmoid_scale": 1.5,
        "flow.normal_low_snr": True,
        "flow.cabal_shift": True,
        "flow.cabal_shift_offset": 0.25,
        "dataset.resolutions": [512, 1024],
    }
    path = tmp_path / "recipe.toml"
    bridge.write_toml(path, flat)
    cfg = load_config(path)
    assert cfg.flow.logit_mean == -0.2 and cfg.flow.sigmoid_scale == 1.5
    assert (
        cfg.flow.cabal_shift
        and cfg.flow.normal_low_snr
        and cfg.flow.cabal_shift_offset == 0.25
    )
    bridge.write_toml(
        path,
        flat
        | {
            "flow.shift": 1,
            "flow.cabal_shift_offset": 0,
            "dataset.resolutions": [256, 512],
        },
    )
    with pytest.raises(ValueError, match="Cabal shift must be positive"):
        load_config(path)


def test_qt_controls_and_two_dimensional_preview(monkeypatch, tmp_path):
    from PySide6 import QtWidgets
    from trainer.gui.app import TrainingGUI
    from trainer.gui import bridge
    from trainer.gui.preferences import Preferences

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(TrainingGUI, "_load_presets", lambda self, **kw: None)
    gui = TrainingGUI(
        preferences=Preferences(tmp_path / "gui.json"), config_dir=str(tmp_path)
    )
    try:
        gui._apply(
            bridge.defaults()
            | {"dataset.path": "/images", "train.model_family": "moss_image"}
        )
        gui.editors["flow.logit_mean"].set(-0.2)
        gui.editors["flow.sigmoid_scale"].set(1.5)
        gui.editors["flow.cabal_shift"].set(True)
        gui._on_edit()
        assert gui.collect()["flow.cabal_shift"]
        gui.editors["flow.flux_shift"].set(True)
        gui._on_edit()
        assert (
            not gui.collect()["flow.cabal_shift"]
            and gui.collect()["flow.shift"] is None
        )
        gui.editors["flow.normal_low_snr"].set(True)
        gui.editors["flow.timestep_sample_method"].set("moss")
        gui._on_edit()
        assert gui.collect()["flow.normal_low_snr"]
        assert not gui.editors["flow.cabal_shift"].widget.isEnabled()
        assert not gui.editors["flow.logit_mean"].widget.isEnabled()
        from trainer.gui.timestep_plot import TimestepDistributionWidget

        plot = TimestepDistributionWidget()
        plot.set_config(bridge.defaults() | {"flow.cabal_shift": True})
        assert plot.canvas.result.shift == 6 and not plot.canvas.error
        plot.deleteLater()
    finally:
        gui.close()
        gui.deleteLater()
        app.processEvents()


def test_new_sampler_rng_resume_and_preview_isolation(tmp_path):
    from trainer.training.flow_rng import FlowRNG

    cfg = FlowConfig(
        logit_mean=-0.2, sigmoid_scale=1.5, normal_low_snr=True, cabal_shift=True
    )
    rng = FlowRNG(19, 0, 1, "cpu")
    sample_timesteps(cfg, 100, 32, 32, "cpu", generator=rng.timesteps)
    rng.save(tmp_path)
    before = torch.get_rng_state().clone()
    distribution(cfg, 32, 32)
    assert torch.equal(before, torch.get_rng_state())
    expected = sample_timesteps(cfg, 100, 32, 32, "cpu", generator=rng.timesteps)
    restored = FlowRNG(200, 0, 1, "cpu")
    restored.load(tmp_path, required=True)
    actual = sample_timesteps(cfg, 100, 32, 32, "cpu", generator=restored.timesteps)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("mode", ["cabal", "flux"])
def test_dynamic_shift_applies_to_each_actual_image_shape(mode):
    from trainer.training.flow import apply_flux_shift, prepare_training_flow_batch

    cfg = FlowConfig(
        logit_mean=-0.2,
        sigmoid_scale=1.5,
        cabal_shift=mode == "cabal",
        flux_shift=mode == "flux",
        shift=6 if mode == "cabal" else None,
    )
    for h, w, reference in [(32, 32, 2.75), (48, 48, 4.25), (64, 64, 6), (32, 128, 6)]:
        seed = 32 + h + w
        base = torch.sigmoid(
            torch.randn(128, generator=torch.Generator().manual_seed(seed)) * 1.5 - 0.2
        )
        expected = (
            apply_static_shift(base, reference)
            if mode == "cabal"
            else apply_flux_shift(base, h, w)
        )
        actual = sample_timesteps(
            cfg, 128, h, w, "cpu", generator=torch.Generator().manual_seed(seed)
        )
        assert torch.equal(actual, expected)
        result = prepare_training_flow_batch(
            torch.zeros(1, 1, 1, h, w), cfg, quantile=0.5
        )
        factor = reference if mode == "cabal" else distribution(cfg, h, w).shift
        median = apply_static_shift(torch.sigmoid(torch.tensor(-0.2)), factor)
        assert result[1].item() == pytest.approx(median.item())


def test_sampling_settings_survive_checkpoint_metadata(tmp_path):
    from trainer.gui import bridge
    from trainer.training.config import load_config
    from trainer.modeling.checkpoint_metadata import encode_settings
    from trainer.tools.recover_config import recover_file
    from safetensors.torch import save_file

    flat = bridge.defaults() | {
        "dataset.path": "/images",
        "train.model_family": "moss_image",
        "flow.logit_mean": -0.2,
        "flow.sigmoid_scale": 1.5,
        "flow.normal_low_snr": True,
        "flow.cabal_shift": True,
        "flow.cabal_shift_offset": 0.25,
    }
    config = tmp_path / "config.toml"
    bridge.write_toml(config, flat)
    original = load_config(config)
    checkpoint = tmp_path / "weights.safetensors"
    save_file(
        {"weight": torch.ones(1)},
        checkpoint,
        metadata={"training_config": encode_settings(original)},
    )
    recovered = tmp_path / "recovered.toml"
    recover_file(checkpoint, recovered)
    assert asdict(load_config(recovered).flow) == asdict(original.flow)


def test_dynamic_preview_uses_configured_resolutions_without_manual_grid_dependency():
    from trainer.gui import bridge
    from trainer.ui_core.config import evaluate

    flat = bridge.defaults() | {
        "dataset.path": "/images",
        "flow.cabal_shift": True,
        "flow.shift": 1,
        "flow.cabal_shift_offset": -1,
        "dataset.resolutions": [2048],
    }
    preview = evaluate(flat)
    assert preview["valid"]
    assert (
        preview["distribution"] is None
    )  # The separate 1024px manual 2D grid is invalid.
    assert [curve["resolution"] for curve in preview["resolutionDistributions"]] == [
        2048
    ]
    assert preview["resolutionDistributions"][0]["shift"] == 3.5
