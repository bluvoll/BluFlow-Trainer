"""Exposure accounting and fractional epoch execution use one resolved plan."""
import copy
import math
import sys

import pytest
import torch

from trainer.training.resolution_schedule import ResolutionSchedule
from trainer.training.stage_plan import resolve_stage_plan
from trainer.data.staged import StagedBatchSampler
from test_resolution_schedule import Samples, make_trainer, enable_checkpoint_saving, config_text


def explicit_plan():
    return ResolutionSchedule(enabled=True, unit='stage_epochs', stages=[
        dict(resolution=r, epochs=e, lr_mul=lr)
        for r, e, lr in [(512, .5, .8), (768, 1.5, .9), (1024, 2.25, 1.)]])


def fractional_trainer(root, kind='lora', workers=0, restart_lr=False):
    def setup(t):
        t.cfg.resolution_schedule = explicit_plan()
        t.cfg.resolution_schedule.restart_lr_each_stage = restart_lr
        if restart_lr:
            t.cfg.schedule.warmup_steps = 2
        t.cfg.train.max_steps = None
        t.sampler = StagedBatchSampler(t.dataset, t.cfg.train.batch_size, t.cfg.resolution_schedule,
                                      None, world=t.accelerator.num_processes,
                                      rank=t.accelerator.process_index, accumulation=2, seed=42)
        t.total_steps = sum(t.sampler.budgets)
    return make_trainer(root, kind=kind, workers=workers, accumulation=2, epoch_mode=True, data_setup=setup)


def test_kuse_exposure_and_epoch_ddp_rounding():
    tiers = [512, 768, 1024, 1280, 1536]
    plan = ResolutionSchedule(enabled=True, stages=[dict(resolution=r, share=s)
                             for r, s in zip(tiers, [.1, .1, .4, .2, .2])])
    batches = dict(zip(tiers, [28, 13, 7, 3, 3]))
    rows = resolve_stage_plan(plan, 900, dict.fromkeys(tiers, 193), batches, 2)
    assert [r['images'] for r in rows] == [5040, 2340, 5040, 1080, 1080]
    assert rows[0]['equivalent_epochs'] == pytest.approx(26.1139896)
    plan = ResolutionSchedule(enabled=True, unit='stage_epochs', stages=[dict(resolution=512, epochs=1)])
    row = resolve_stage_plan(plan, None, {512: 193}, {512: 28}, 2)[0]
    assert (row['steps'], row['images'], row['padding']) == (4, 224, 31)
    assert row['warnings']


@pytest.mark.parametrize('duration', [0, -1, True, math.nan, math.inf])
def test_bad_explicit_duration(duration):
    with pytest.raises(ValueError):
        ResolutionSchedule(enabled=True, unit='stage_epochs', stages=[dict(resolution=512, epochs=duration)])


def test_ambiguous_durations_rejected():
    for unit in ('steps', 'epochs', 'stage_epochs'):
        with pytest.raises(ValueError):
            ResolutionSchedule(enabled=True, unit=unit, stages=[dict(resolution=512, share=1, epochs=2)])


@pytest.mark.parametrize('world', [1, 2])
def test_fractional_plan_matches_consumption_and_resume_cursor(tmp_path, world):
    samples = Samples(tmp_path)
    for stage in range(3):
        seen = 0
        for rank in range(world):
            sampler = StagedBatchSampler(samples, {512: 3, 768: 2, 1024: 1}, explicit_plan(), None,
                                        world=world, rank=rank, accumulation=2)
            first = sum(sampler.epoch_counts[:stage])
            for epoch in range(first, first + sampler.epoch_counts[stage]):
                sampler.configure_epoch(stage, epoch)
                full = list(sampler)
                seen += sum(map(len, full))
                for cursor in range(len(full) + 1):
                    sampler.configure_epoch(stage, epoch, cursor)
                    assert list(sampler) == full[cursor:]
        assert seen == sampler.plan[stage]['images']
    assert sampler.budgets == ([1, 3, 9] if world == 1 else [1, 2, 5])


@pytest.mark.parametrize('kind', ['none', 'lora', 'lycoris_lora', 'lokr', 'lycoris_lokr'])
def test_fractional_loop_and_exact_resume(tmp_path, kind, monkeypatch):
    monkeypatch.setenv('LYCORIS_KERNEL_BACKEND', 'torch')
    trainer = fractional_trainer(tmp_path, kind)
    enable_checkpoint_saving(trainer, tmp_path)
    trainer.train()
    assert trainer.stage_images == [4, 11, 16]
    assert trainer.global_step == 13
    assert trainer.stage_attempts == [1, 3, 9]
    assert trainer._resolution_state()['plan'] == trainer.sampler.plan
    final = {k: v.detach().clone() for k, v in trainer._unwrap(trainer.transformer).state_dict().items()}
    trace, lrs = trainer.trace, trainer.lr_trace
    trainer.accelerator.free_memory()
    # Inside a full epoch, followed by the partial traversal and a new stage.
    resumed = fractional_trainer(tmp_path, kind)
    enable_checkpoint_saving(resumed, tmp_path)
    resumed._resume(str(tmp_path / 'output' / 'staged-test-step000002-state'))
    resumed.train()
    assert resumed.trace == trace[4:]
    assert resumed.lr_trace == lrs[2:]
    assert resumed.stage_images == [4, 11, 16]
    for key, value in resumed._unwrap(resumed.transformer).state_dict().items():
        torch.testing.assert_close(value, final[key], rtol=0, atol=0)
    resumed.accelerator.free_memory()


def test_explicit_roundtrip_and_ui_transition(tmp_path):
    from trainer.gui import bridge
    from trainer.ui_core.config import normalize, evaluate
    from trainer.training.config import load_config
    from dataclasses import asdict
    flat = bridge.defaults() | {
        'train.model_family': 'moss_image', 'adapter.kind': 'lora', 'dataset.path': str(tmp_path),
        'dataset.source': 'latents', 'dataset.resolutions': [512, 768, 1024],
        'train.pack_resolutions': True, 'train.batch_size': {512: 3, 768: 2, 1024: 1},
        'train.attention_backend': 'flash_attn_2' if sys.platform == 'win32' else 'torch_varlen',
        'resolution_schedule.enabled': True, 'resolution_schedule.unit': 'stage_epochs',
        'resolution_schedule.stages': [{k: v for k, v in asdict(s).items() if v is not None} for s in explicit_plan().stages],
    }
    path = tmp_path / 'explicit.toml'
    bridge.write_toml(path, flat)
    cfg = load_config(path)
    assert cfg.resolution_schedule.budgets(None) == [.5, 1.5, 2.25]
    assert cfg.train.max_steps is None
    state = evaluate(flat)
    assert state['valid'], state['error']
    notes = ' '.join(message for _, message in bridge.advisories(flat, 2))
    assert 'requested stage epochs fixed' in notes
    assert 'Multiply epochs' not in notes
    assert [s['epochs'] for s in state['stages']] == [.5, 1.5, 2.25]
    shares = normalize(flat | {'resolution_schedule.unit': 'epochs'}, flat)
    assert sum(s['share'] for s in shares['resolution_schedule.stages']) == pytest.approx(1)
    assert all('epochs' not in s for s in shares['resolution_schedule.stages'])
    # Electron can batch a unit switch and an immediate row edit before the
    # backend has replaced the old share rows. Preserve the new duration.
    draft = copy.deepcopy(shares)
    draft['resolution_schedule.unit'] = 'stage_epochs'
    draft['resolution_schedule.stages'][0]['epochs'] = .75
    converted = normalize(draft, shares)
    assert converted['resolution_schedule.stages'][0]['epochs'] == .75
    assert all('share' not in s for s in converted['resolution_schedule.stages'])
