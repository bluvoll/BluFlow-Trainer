"""Opt-in LoKr probes over frozen INT8 projections, with and without rotation."""
import copy
import os

import pytest
import torch

from trainer.modeling.moss_image import MOSSImage, MOSSParams
from trainer.training.block_lr import BlockLRConfig
from trainer.training.config import Config, OptimizerConfig
from trainer.training.params import AdapterConfig, ComponentLRs, apply_adapter, build_adapter_param_groups
from trainer.training.quant import QuantConfig, quantize_module
from trainer.training.optim import build_optimizer
from trainer.training.distributed import model_storage
from test_sdnq_finetune_matmul import step, snapshot, assert_same

pytestmark = pytest.mark.skipif(os.environ.get('BLUFLOW_GPU_TESTS') != '1', reason='GPU opt-in')


def make_model(device, backend, preset, dense, compiled=True, initialization='lycoris'):
    torch.manual_seed(321)
    model = MOSSImage(MOSSParams(dim=256, ffn_dim=512, num_heads=4,
                                num_layers=6, text_dim=64, latent_channels=8)).to(device, torch.bfloat16)
    quant = QuantConfig(preset=preset)
    quantize_module(model, quant, device, torch.bfloat16, True)
    from trainer.training.frozen_matmul import FrozenINT8Linear
    layers = [m for m in model.modules() if isinstance(m, FrozenINT8Linear)]
    assert layers
    assert all((m._sdnq_rotation is not None) == (preset == 'balanced') for m in layers)
    rank, factor = (10000, 1) if dense else (8, 4)
    adapter = AdapterConfig(kind='lokr' if backend == 'peft' else 'lycoris_lora',
                            lycoris_algo='lokr', lycoris_bypass=True,
                            rank=rank, alpha=rank / 2, lokr_factor=factor,
                            lokr_init=initialization, dtype='bfloat16', moss_train_text_refiner=False)
    apply_adapter(model, adapter)
    block = BlockLRConfig(True, [dict(start=1, end=1, lr_mul=0),
                                 dict(start=2, end=2, lr_mul=.75), dict(start=3, end=3, lr_mul=1.5)])
    groups = build_adapter_param_groups(model, ComponentLRs(), 1e-4, weight_decay=.01, block_lr=block)
    opt = build_optimizer(groups.groups, OptimizerConfig(kind='came', lr=1e-4))
    model.configure_execution(True, compile_mode='default' if compiled else None, attention_backend='torch_varlen')
    return model, opt, adapter, quant


@pytest.mark.parametrize('backend', ['peft', 'lycoris'])
@pytest.mark.parametrize('preset', ['balanced', 'fast'])
@pytest.mark.parametrize('dense', [False, True])
def test_lokr_compiled_update_export_resume(tmp_path, backend, preset, dense):
    from trainer.modeling.export import export_checkpoint
    from trainer.training.adapter_loading import load_adapter_weights
    from trainer.data.dataset import DatasetConfig
    from safetensors.torch import load_file
    device = torch.device('cuda')
    model, opt, adapter, quant = make_model(device, backend, preset, dense)
    initial = snapshot(model)
    frozen_names = {n for n, p in model.named_parameters() if not p.requires_grad}
    frozen_names |= {n for n, _ in model.named_buffers()}
    losses = [step(model, opt, device, i) for i in range(2)]
    current = snapshot(model)
    assert_same({n: current[n] for n in frozen_names}, {n: initial[n] for n in frozen_names})
    assert any(not torch.equal(t, current[n]) for n, t in initial.items() if '.lokr_' in n)
    saved_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    saved_opt = copy.deepcopy(opt.state_dict())
    expected_loss = step(model, opt, device, 2)
    expected = snapshot(model)
    fresh, newopt, _, _ = make_model(device, backend, preset, dense)
    fresh.load_state_dict(saved_model)
    newopt.load_state_dict(saved_opt)
    assert_same(snapshot(fresh), current)
    assert step(fresh, newopt, device, 2) == expected_loss
    assert_same(snapshot(fresh), expected)
    cfg = Config(dataset=DatasetConfig(path=str(tmp_path)), adapter=adapter, quant=quant)
    cfg.train.model_family = 'moss_image'
    export_checkpoint(fresh, tmp_path, 'probe', cfg, torch.bfloat16, 3)
    path = tmp_path / 'probe.safetensors'
    exported = load_file(path)
    assert exported and all('.lokr_' in k or k.endswith('.alpha') for k in exported)
    assert all(torch.isfinite(t).all() for t in exported.values())
    # Alpha=rank/2 also exercises PEFT's dense export scaling and its inverse.
    with torch.no_grad():
        for n, p in fresh.named_parameters():
            if '.lokr_' in n:
                p.zero_()
    load_adapter_weights(fresh, path)
    assert_same(snapshot(fresh), expected)
    print(f'PASS {backend} {preset} dense={dense}: compiled updates, frozen base/block, export, exact resume; losses={losses+[expected_loss]} peak_GB={torch.cuda.max_memory_allocated()/1e9:.3f}', flush=True)
    torch._dynamo.reset()


def test_peft_default_initialization_hadamard():
    device = torch.device('cuda')
    model, opt, _, _ = make_model(device, 'peft', 'balanced', False, initialization='peft')
    before = snapshot(model)
    for index in range(3):
        step(model, opt, device, index)
    after = snapshot(model)
    assert any(not torch.equal(before[n], t) for n, t in after.items() if '.lokr_' in n)
    assert_same({n: t for n, t in after.items() if '.lokr_' not in n},
                {n: t for n, t in before.items() if '.lokr_' not in n})
    print('PASS PEFT default initialization: compiled Hadamard updates, unchanged base', flush=True)
    torch._dynamo.reset()


@pytest.mark.parametrize('backend', ['peft', 'lycoris'])
def test_lokr_hadamard_ddp(ddp_device, backend):
    import torch.distributed as dist
    from trainer.training.distributed import configure_quantized_ddp, sync_quantized_model
    model, opt, _, _ = make_model(ddp_device, backend, 'balanced', False)
    sync_quantized_model(model)
    rank = dist.get_rank()
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], **configure_quantized_ddp().to_kwargs())
    before = snapshot(model)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    frozen |= {n for n, _ in model.named_buffers()}
    for index in range(3):
        step(ddp, opt, ddp_device, index, rank)
    for name, tensor in model_storage(model):
        reference = tensor.detach().clone().contiguous()
        dist.broadcast(reference, 0)
        torch.testing.assert_close(tensor, reference, rtol=0, atol=0)
        if name in frozen:
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    print(f'PASS {backend} Hadamard DDP rank={rank}: identical replicas and frozen base/block', flush=True)
    torch._dynamo.reset()
