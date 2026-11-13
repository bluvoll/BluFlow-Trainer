"""Real weighted data loading, variable epoch sizes and checkpointed model RNG."""
import copy

import pytest
import torch
from safetensors.torch import save_file

from trainer.data.dataset import DatasetConfig, MageFlowDataset
from trainer.data.cache import cache_path
from trainer.training.train import Trainer
from test_resolution_schedule import make_trainer, enable_checkpoint_saving


class TinyLatentDataset(MageFlowDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['latents'] = sample['latents'][:8]
        return sample


def weighted_trainer(root, monkeypatch, staged=False):
    from trainer.data.bucket import BucketManager
    from trainer.training.resolution_schedule import ResolutionSchedule
    for i in range(9):
        size = (128, 128) if staged else [(128, 128), (128, 64), (64, 128)][i % 3]
        image = root / f'image{i}.png'
        image.with_suffix('.txt').write_text(f'image{i}, a, b, c, d')
        for tier in ([64, 128] if staged else [128]):
            manager = BucketManager(no_upscale=True, max_reso=(tier, tier), min_size=64, max_size=4096, reso_steps=64)
            bucket = manager.select_bucket(*size)[0]
            path = cache_path(image, bucket)
            if not path.exists():
                save_file({'latents': torch.full((128, 1, bucket[1]//16, bucket[0]//16), (i+1)/10)},
                          path, metadata={'original_size': f'{size[0]}x{size[1]}'})
    monkeypatch.setattr('trainer.training.train.MageFlowDataset', TinyLatentDataset)
    def setup(t):
        t.cfg.resolution_schedule.enabled = False
        t.cfg.train.pack_resolutions = False
        t.cfg.train.epochs = 4
        t.cfg.train.max_steps = None
        t.cfg.train.batch_size = 3
        t.cfg.dataset = DatasetConfig(path=str(root), source='latents', resolution=128,
                                      min_bucket_reso=64, num_repeats=1.56, cache_validation='off')
        if staged:
            t.cfg.resolution_schedule = ResolutionSchedule(enabled=True, unit='epochs', stages=[
                dict(resolution=64, share=.5), dict(resolution=128, share=.5)])
            t.cfg.dataset.resolutions = [64, 128]
            t.cfg.train.pack_resolutions = True
            t.cfg.train.attention_backend = 'flash_attn_2'
            t.cfg.train.batch_size = {64: 2, 128: 3}
        Trainer._build_data(t)
        if not staged:
            t.loader = t.accelerator.prepare_data_loader(t.loader)
    t = make_trainer(root, data_setup=setup)
    t.phase = None
    original = t._step
    def stochastic_step(batch):
        loss, hf = original(batch)
        return loss * torch.rand(()), hf
    t._step = stochastic_step
    return t


@pytest.mark.parametrize('staged,checkpoint_step', [(False, 1), (False, 3), (False, 4), (True, 3)])
def test_weighted_epoch_full_state_resume(tmp_path, monkeypatch, staged, checkpoint_step):
    t = weighted_trainer(tmp_path, monkeypatch, staged)
    enable_checkpoint_saving(t, tmp_path)
    t.train()
    assert not t.accelerator.gradient_state.in_dataloader
    final = copy.deepcopy(t._unwrap(t.transformer).state_dict())
    trace, lr = list(t.trace), list(t.lr_trace)
    import json
    checkpoint = tmp_path / 'output' / f'staged-test-step{checkpoint_step:06d}-state'
    state = json.loads((checkpoint/'state.json').read_text())['resolution_schedule_state' if staged else 'sampling_state']
    consumed = 0  # cursor is rank-local, without prefetch
    if staged:
        consumed = sum(n * b for n, b in zip(t.sampler.epoch_budgets[:state['stage']],
                                           t.sampler.epoch_batches[:state['stage']]))
        local_epoch = state['epoch'] - sum(t.sampler.epoch_budgets[:state['stage']])
        consumed += local_epoch * t.sampler.epoch_batches[state['stage']]
    else:
        for epoch in range(state['epoch']):
            t.sampler.set_epoch(epoch)
            consumed += len(t.sampler)
    consumed += state['cursor']
    t.accelerator.free_memory()
    resumed = weighted_trainer(tmp_path, monkeypatch, staged)
    enable_checkpoint_saving(resumed, tmp_path)
    resumed._resume(str(checkpoint))
    resumed.train()
    assert resumed.trace == trace[consumed:]
    assert resumed.lr_trace == lr[checkpoint_step:]
    for key, value in resumed._unwrap(resumed.transformer).state_dict().items():
        torch.testing.assert_close(value, final[key], rtol=0, atol=0)
    resumed.accelerator.free_memory()
