"""Finite staged epochs, partial accumulation, and exact checkpoint continuation."""
import copy

import pytest
import torch

from trainer.data.staged import StagedBatchSampler
from test_resolution_schedule import Samples, plan, make_trainer, enable_checkpoint_saving, config_text


def epoch_trainer(root, workers=0):
    t = make_trainer(root, workers=workers, accumulation=2, epoch_mode=True)
    t.cfg.train.max_steps = None
    t.cfg.train.save_every_epochs = 1
    return t


def test_epoch_allocation():
    p = plan()
    p.unit = 'epochs'
    assert p.budgets(50) == [10, 10, 30]
    assert p.budgets(7) == [2, 1, 4]
    with pytest.raises(ValueError, match='at least one epoch'):
        p.budgets(2)


def test_epoch_config(tmp_path):
    import sys
    from trainer.training.config import load_config
    text = config_text().replace('enabled = true', 'enabled = true\nunit = "epochs"')
    text = text.replace('max_steps = 560', 'epochs = 50\nsave_every_epochs = 2')
    if sys.platform == 'win32':
        text = text.replace('torch_varlen', 'flash_attn_2')
    path = tmp_path / 'epochs.toml'
    path.write_text(text)
    cfg = load_config(path)
    assert cfg.train.max_steps is None
    assert cfg.train.save_every_epochs == 2
    assert cfg.resolution_schedule.budgets(cfg.train.epochs) == [10, 10, 30]


def test_finite_epoch_rank_padding_and_cursor(tmp_path):
    ds = Samples(tmp_path, 7)
    p = plan()
    p.unit = 'epochs'
    ranks = []
    for rank in (0, 1):
        sampler = StagedBatchSampler(ds, {512: 2, 768: 2, 1024: 2}, p, 5,
                                    world=2, rank=rank, accumulation=3)
        sampler.configure_epoch(0, 0)
        batches = list(sampler)
        assert len(batches) == 2
        sampler.configure_epoch(0, 0, 1)
        assert list(sampler) == batches[1:]
        ranks.extend(s[0] for b in batches for s in b)
    assert set(ranks) == set(range(7))
    assert len(ranks) == 8  # only the final distributed microbatch is padded


@pytest.mark.parametrize('workers', [0, 2])
def test_epoch_loop_and_partial_accumulation(tmp_path, workers):
    t = epoch_trainer(tmp_path, workers)
    t.train()
    assert t.sampler.epoch_budgets == [1, 1, 3]
    assert t.sampler.epoch_batches == [3, 4, 7]
    assert t.sampler.budgets == [2, 2, 12]
    assert t.global_step == t.successful_updates == 16
    assert t.stage_attempts == [2, 2, 12]
    assert t.stage_images == [7, 7, 21]
    assert not t.accelerator.gradient_state.in_dataloader
    assert [s for s, _ in t.trace] == [0]*3 + [1]*4 + [2]*21
    assert [tag for tag, _ in t.saved if tag.startswith('epoch')] == [f'epoch{i:03}' for i in range(1, 6)]
    t.accelerator.free_memory()


def test_skipped_update_does_not_extend_epoch(tmp_path):
    t = epoch_trainer(tmp_path)
    step = t._step
    def bad_once(batch):
        loss, hf = step(batch)
        return (loss * float('nan') if len(t.trace) == 1 else loss), hf
    t._step = bad_once
    t.train()
    assert len(t.trace) == 28
    assert t.global_step == 16
    assert t.successful_updates == 15
    assert t.skipped_updates == 1
    t.accelerator.free_memory()


@pytest.mark.parametrize('checkpoint_step', [1, 2, 4, 5])
def test_epoch_full_state_resume(tmp_path, checkpoint_step):
    t = epoch_trainer(tmp_path)
    enable_checkpoint_saving(t, tmp_path)
    t.train()
    final = copy.deepcopy(t._unwrap(t.transformer).state_dict())
    trace, lr = list(t.trace), list(t.lr_trace)
    # Update 2 ends a partial group; update 4 is also a stage boundary.
    consumed = {1: 2, 2: 3, 4: 7, 5: 9}[checkpoint_step]
    t.accelerator.free_memory()
    resumed = epoch_trainer(tmp_path)
    enable_checkpoint_saving(resumed, tmp_path)
    resumed._resume(str(tmp_path / 'output' / f'staged-test-step{checkpoint_step:06d}-state'))
    resumed.train()
    assert resumed.trace == trace[consumed:]
    assert resumed.lr_trace == lr[checkpoint_step:]
    assert resumed.global_step == 16
    for key, value in resumed._unwrap(resumed.transformer).state_dict().items():
        torch.testing.assert_close(value, final[key], rtol=0, atol=0)
    resumed.accelerator.free_memory()
