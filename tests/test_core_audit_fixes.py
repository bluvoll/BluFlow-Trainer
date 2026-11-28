"""Regression tests for core training-loop audit fixes."""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch

from trainer.training.config import FlowConfig, ScheduleConfig
from trainer.training.flow import flow_loss
from trainer.training.optim import StageLRScheduler, build_scheduler


@pytest.mark.parametrize("kind", ["rex", "rerex", "cosine", "linear", "constant"])
@pytest.mark.parametrize("world", [2, 4])
def test_ddp_scheduler_matches_single_gpu_per_update(kind, world):
    def make_opt():
        return torch.optim.SGD([{"params": [torch.nn.Parameter(torch.ones(1))], "lr": 1e-3}])

    cfg = ScheduleConfig(kind=kind, warmup_steps=5, min_lr_ratio=0.05)
    total_updates = 20

    opt1 = make_opt()
    sched1 = build_scheduler(opt1, cfg, total_updates, ticks_per_update=1)

    optN = make_opt()
    schedN = build_scheduler(optN, cfg, total_updates, ticks_per_update=world)

    for _ in range(total_updates):
        assert optN.param_groups[0]["lr"] == pytest.approx(opt1.param_groups[0]["lr"])
        opt1.step()
        sched1.step()
        optN.step()
        for _ in range(world):
            schedN.step()


def test_ddp_stage_scheduler_first_step_warmup_not_divided_by_world():
    opt = torch.optim.SGD([{"params": [torch.nn.Parameter(torch.ones(1))], "lr": 1e-3}])
    cfg = ScheduleConfig(kind="rerex", warmup_steps=5, min_lr_ratio=0.05)
    sched = StageLRScheduler(opt, cfg, budgets=[3, 12], ticks_per_update=2)
    # Stage 0 has 3 updates -> stage_warmup_steps(5, 3) == 1 -> step 0 should be full peak 1e-3, not 0.5e-3.
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    for _ in range(3):
        opt.step()
        sched.step()
        sched.step()
    sched.begin_stage(1)
    # Stage 1 has warmup=5 -> step 0 should be 1/5 * 1e-3, not 1/10 * 1e-3.
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3 / 5.0)


def test_flow_loss_masked_normalizes_per_sample():
    # Sample 0 has 2 valid pixels with squared error 1.0 -> per-sample mean = 1.0
    # Sample 1 has 8 valid pixels with squared error 9.0 -> per-sample mean = 9.0
    # Equal per-sample weighting gives (1.0 + 9.0) / 2 = 5.0 (pooled would give 7.4).
    pred = torch.zeros(2, 1, 2, 4)
    target = torch.zeros(2, 1, 2, 4)
    target[0] = 1.0
    target[1] = 3.0
    mask = torch.zeros(2, 1, 2, 4)
    mask[0, :, 0, :2] = 1.0
    mask[1, :, :, :] = 1.0
    loss = flow_loss(pred, target, mask=mask)
    assert float(loss) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timestep_sample_method": "moss", "shift": 1.0, "logit_mean": -0.2},
        {"timestep_sample_method": "uniform", "logit_mean": -0.2},
    ],
)
def test_flow_config_warns_when_logit_mean_ignored(kwargs):
    with pytest.warns(UserWarning, match="logit_mean"):
        FlowConfig(**kwargs)


def test_flow_config_no_warning_for_default_logit_mean_or_logit_normal():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FlowConfig(timestep_sample_method="moss", shift=1.0, logit_mean=0.0)
        FlowConfig(timestep_sample_method="logit_normal", logit_mean=-0.2)
    assert not [w for w in caught if "logit_mean" in str(w.message)]


def test_trainer_log_handles_missing_last_update_lrs():
    from trainer.training.train import Trainer

    t = object.__new__(Trainer)
    t._last_update_lrs = None
    t.scheduler = SimpleNamespace(get_last_lr=lambda: [1e-4])
    t.accelerator = SimpleNamespace(is_main_process=True, log=lambda *a, **kw: None)
    t.global_step = 0
    t.bar = None
    t.loss_window = []
    t.cfg = SimpleNamespace(train=SimpleNamespace(log_every=1))
    t.timing = SimpleNamespace(summary=lambda: {})
    t.skipped_updates = 0
    t.successful_updates = 0
    t.consecutive_nonfinite = 0
    t.resolution_schedule = None
    # Should not raise AttributeError / TypeError when _last_update_lrs is None.
    assert (t._last_update_lrs if t._last_update_lrs is not None else t.scheduler.get_last_lr()) == [1e-4]


def test_step_safety_foreach_and_clip_with_norm():
    import torch
    from trainer.training.step_safety import (
        clip_gradients_with_norm,
        global_sample_scale,
        gradient_norm,
        scale_gradients,
    )

    p1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p2 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p1.grad = torch.tensor([3.0, 4.0])
    p2.grad = torch.tensor([0.0, 12.0])

    norm = gradient_norm([p1, p2], torch.device("cpu"))
    assert abs(norm.item() - 13.0) < 1e-5

    # No-op when max_grad_norm >= total_norm or max_grad_norm <= 0
    clip_gradients_with_norm([p1, p2], 15.0, norm)
    assert abs(gradient_norm([p1, p2], torch.device("cpu")).item() - 13.0) < 1e-5
    clip_gradients_with_norm([p1, p2], 0.0, norm)
    assert abs(gradient_norm([p1, p2], torch.device("cpu")).item() - 13.0) < 1e-5

    # Clip when total_norm > max_grad_norm
    clip_gradients_with_norm([p1, p2], 6.5, norm)
    assert abs(gradient_norm([p1, p2], torch.device("cpu")).item() - 6.5) < 1e-5

    scale_gradients([p1, p2], 2.0)
    assert abs(gradient_norm([p1, p2], torch.device("cpu")).item() - 13.0) < 1e-5

    # Fast path skips `.item()` sync check when local_samples > 0 and returns `world * accumulation_steps / count`.
    scale = global_sample_scale(3, 2, torch.device("cpu"))
    assert abs(scale.item() - (2.0 / 3.0)) < 1e-6


def test_curriculum_phase_lr_mul_logged(capsys):
    import torch
    from trainer.training.train import Trainer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = object.__new__(Trainer)
    t.accelerator = SimpleNamespace(
        is_main_process=True, device=dev, num_processes=1, gather=lambda x: x
    )
    t.progress_mode = "plain"
    t.cfg = SimpleNamespace(
        resolution_schedule=SimpleNamespace(enabled=False),
        train=SimpleNamespace(log_every=1),
        optimizer=SimpleNamespace(max_grad_norm=0.0),
    )
    t.phase = SimpleNamespace(lr_mul=0.5)
    t._last_update_lrs = [5e-5]
    t.scheduler = SimpleNamespace(get_last_lr=lambda: [1e-4])
    t.global_step = 1
    t.total_steps = 10
    t.bar = None
    t.last_preserve = None
    t.last_grad_norm = None
    t.last_stats = {}
    t._eta = lambda: ""

    t._log(torch.tensor(0.25), None, epoch=0, t0=0.0)
    assert "lr 5.00e-05" in capsys.readouterr().out

    t.phase = SimpleNamespace(lr_mul=1.0)
    t._last_update_lrs = None
    t._log(torch.tensor(0.25), None, epoch=0, t0=0.0)
    assert "lr 1.00e-04" in capsys.readouterr().out
