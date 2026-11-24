"""Two CPU ranks: finite epochs, partial accumulation and exact continuation."""
from datetime import timedelta
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest


def epoch_worker(rank, root, explicit=False, kind='lora', restart_lr=False):
    from test_resolution_schedule import Samples, enable_checkpoint_saving
    from test_staged_epochs import epoch_trainer
    from test_stage_exposure import fractional_trainer
    factory = fractional_trainer if explicit else epoch_trainer
    options = dict(kind=kind, restart_lr=True) if restart_lr else dict(kind=kind)
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE='2', LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE='2',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT='29587', OMP_NUM_THREADS='1')
    root = Path(root)
    dist.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        if rank == 0:
            Samples(root)
        dist.barrier()
        first = factory(root, **options)
        first.accelerator.end_training = lambda: None
        enable_checkpoint_saving(first, root)
        first.train()
        assert first.sampler.epoch_batches == [2, 2, 4]
        assert first.sampler.budgets == ([1, 2, 5] if explicit else [1, 1, 6])
        assert first.global_step == first.successful_updates == 8
        assert first.stage_attempts == first.sampler.budgets
        assert first.stage_images == [row['images'] for row in first.sampler.plan]
        if restart_lr:
            # Last tier: two warmup updates and two scheduler ticks per update.
            rate = first.cfg.optimizer.lr
            assert [groups[0] for groups in first.lr_trace[3:6]] == pytest.approx([rate / 4, rate * .75, rate])
        final = {k: v.detach().clone() for k, v in first._unwrap(first.transformer).state_dict().items()}
        trace = list(first.trace)
        for value in final.values():
            reference = value.clone()
            dist.broadcast(reference, 0)
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        first.accelerator.free_memory()
        resumed = factory(root, **options)
        resumed.accelerator.end_training = lambda: None
        enable_checkpoint_saving(resumed, root)
        resumed._resume(str(root/'output'/'staged-test-step000003-state'))
        resumed.train()
        assert resumed.trace == trace[4 if explicit else 6:]
        assert resumed.stage_images == first.stage_images
        assert resumed.lr_trace == first.lr_trace[3:]
        for key, value in resumed._unwrap(resumed.transformer).state_dict().items():
            torch.testing.assert_close(value, final[key], rtol=0, atol=0)
        resumed.accelerator.free_memory()
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture
def gloo_device():
    try:
        dist.ProcessGroupGloo.create_default_device()
    except RuntimeError as exc:
        if 'unsupported gloo device' in str(exc):
            pytest.skip(f'This Torch build cannot create a Gloo transport: {exc}')
        raise


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('kind', ['lora', 'none', 'lokr', 'lycoris_lokr'])
def test_two_rank_epochs_and_resume(tmp_path, gloo_device, explicit, kind):
    mp.spawn(epoch_worker, args=(str(tmp_path), explicit, kind), nprocs=2, join=True)


def test_two_rank_stage_lr_restart_and_resume(tmp_path, gloo_device):
    mp.spawn(epoch_worker, args=(str(tmp_path), True, 'lora', True), nprocs=2, join=True)
