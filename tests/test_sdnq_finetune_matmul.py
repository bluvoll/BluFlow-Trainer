"""Opt-in CUDA probes for trainable INT8 matmul, Hadamard, and quantized resume."""
import io
import os

import pytest
import torch

from trainer.modeling.moss_image import MOSSImage, MOSSParams
from trainer.training.block_lr import BlockLRConfig
from trainer.training.config import Config, OptimizerConfig
from trainer.training.distributed import model_storage, quantized_optimizer_rng
from trainer.training.optim import build_optimizer
from trainer.training.params import AdapterConfig, ComponentLRs, build_param_groups
from trainer.training.quant import QuantConfig, quantize_module

pytestmark = pytest.mark.skipif(os.environ.get('BLUFLOW_GPU_TESTS') != '1', reason='GPU opt-in')


def make_model(device, hadamard, matmul=True, compiled=False, kahan=False):
    torch.manual_seed(42)
    cfg = QuantConfig(mode='training', weights_dtype='int8', group_size=-1,
                      use_quantized_matmul=matmul, use_hadamard=hadamard,
                      use_stochastic_rounding=True)
    model = MOSSImage(MOSSParams(dim=256, ffn_dim=512, num_heads=4,
                                num_layers=6, text_dim=64, latent_channels=8)).to(device, torch.bfloat16)
    quantize_module(model, cfg, device, torch.bfloat16, matmul)
    from sdnq.training import SDNQTensor
    quantized = [(n,p) for n,p in model.named_parameters() if isinstance(p,SDNQTensor)]
    assert quantized, 'Probe must actually exercise quantized layers'
    assert all(p.sdnq_dequantizer.use_hadamard == hadamard for _,p in quantized)
    block = BlockLRConfig(True,[dict(start=1,end=1,lr_mul=0),dict(start=2,end=2,lr_mul=.75),dict(start=3,end=3,lr_mul=1.5)])
    groups = build_param_groups(model,ComponentLRs(),1e-4,weight_decay=.01,block_lr=block)
    opt = build_optimizer(groups.groups,OptimizerConfig(kind='adamw8bit',lr=1e-4,use_kahan=kahan))
    model.configure_execution(True,compile_mode='default' if compiled else None,attention_backend='torch_varlen')
    return model,opt,cfg,block


def step(model,opt,device,index,rank=0):
    torch.manual_seed(1000+rank+index*13)
    # Different packed shapes across updates stand in for tier transitions.
    shapes = [(4+index%2,4),(3,5+index%2)]
    xs = [torch.randn(1,8,1,h,w,device=device,dtype=torch.bfloat16) for h,w in shapes]
    text = torch.randn(2,7,64,device=device,dtype=torch.bfloat16)
    mask = torch.ones(2,7,device=device,dtype=torch.bool);mask[1,4:]=False
    outputs = model(xs,torch.tensor([.2,.7],device=device),(text,mask))[0]
    loss = sum(y.float().square().mean() for y in outputs)
    assert torch.isfinite(loss)
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    with quantized_optimizer_rng(device,42,index): opt.step()
    opt.zero_grad(set_to_none=True)
    return float(loss.detach())


def snapshot(model):
    return {n:t.detach().clone() for n,t in model_storage(model)}


def assert_same(actual,expected):
    assert actual.keys()==expected.keys()
    for n,t in actual.items():
        torch.testing.assert_close(t,expected[n],rtol=0,atol=0,msg=lambda msg:n+'\n'+msg)


@pytest.mark.parametrize('matmul,hadamard',[(False,False),(True,False),(True,True)])
@pytest.mark.parametrize('compiled,kahan',[(False,False),(True,True)])
def test_full_moss_updates_freeze_export_resume(tmp_path,matmul,hadamard,compiled,kahan):
    device=torch.device('cuda')
    model,opt,quant,block=make_model(device,hadamard,matmul,compiled,kahan)
    initial=snapshot(model)
    losses=[step(model,opt,device,i) for i in range(2)]
    current=snapshot(model)
    assert any(not torch.equal(t,current[n]) for n,t in initial.items() if n.startswith('layers.2.'))
    assert_same({n:t for n,t in current.items() if n.startswith('layers.1.')},
                {n:t for n,t in initial.items() if n.startswith('layers.1.')})
    # Exercise the native serializer used by Trainer.save_full_state for SDNQTensor.
    buffer=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},buffer)
    loss=step(model,opt,device,2);final=snapshot(model)
    fresh,newopt,_,_=make_model(device,hadamard,matmul,compiled,kahan)
    buffer.seek(0);saved=torch.load(buffer,weights_only=False)
    fresh.load_state_dict(saved['model']);newopt.load_state_dict(saved['optimizer'])
    assert_same(snapshot(fresh),current)
    assert step(fresh,newopt,device,2)==loss
    assert_same(snapshot(fresh),final)
    # Export to ordinary BF16: no SDNQ dependency in the inference checkpoint.
    from trainer.modeling.export import export_checkpoint
    from safetensors.torch import load_file
    from trainer.data.dataset import DatasetConfig
    cfg=Config(dataset=DatasetConfig(path=str(tmp_path)),adapter=AdapterConfig(kind='none'),quant=quant,block_lr=block)
    cfg.train.model_family='moss_image';cfg.train.save_native=True
    export_checkpoint(fresh,tmp_path,'probe',cfg,torch.bfloat16,3)
    exported=load_file(tmp_path/'probe.safetensors')
    # Request separated export for strict reload into the training architecture.
    cfg.train.moss_qkv_format='separated'
    export_checkpoint(fresh,tmp_path,'separated',cfg,torch.bfloat16,3)
    plain=MOSSImage(fresh.params).to(device,torch.bfloat16)
    plain.load_state_dict(load_file(tmp_path/'separated.safetensors'))
    from trainer.training.quant import dequantize_state_dict
    expected=dequantize_state_dict(fresh.state_dict(),torch.bfloat16,device='cpu')
    # RoPE regenerates its analytical FP32 bands on reload, intentionally.
    for n,p in plain.named_parameters():torch.testing.assert_close(p.cpu(),expected[n].cpu(),rtol=0,atol=0,check_dtype=False)
    assert exported
    print(f'PASS matmul={matmul} hadamard={hadamard} compile={compiled} kahan={kahan} losses={losses+[loss]} peak_GB={torch.cuda.max_memory_allocated()/1e9:.3f}',flush=True)
    torch._dynamo.reset()


def test_two_rank_hadamard_updates_and_resume(ddp_device):
    import torch.distributed as dist
    from trainer.training.distributed import configure_quantized_ddp,sync_quantized_model
    rank=dist.get_rank();device=ddp_device
    model,opt,_,_=make_model(device,True,compiled=True,kahan=True)
    sync_quantized_model(model)
    ddp=torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank],**configure_quantized_ddp().to_kwargs())
    for index in range(2):step(ddp,opt,device,index,rank)
    for name,t in model_storage(model):
        ref=t.detach().clone().contiguous();dist.broadcast(ref,0)
        torch.testing.assert_close(t,ref,rtol=0,atol=0)
    current=snapshot(model)
    buffer=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},buffer)
    loss=step(ddp,opt,device,2,rank);final=snapshot(model)
    fresh,newopt,_,_=make_model(device,True,compiled=True,kahan=True)
    sync_quantized_model(fresh)
    newddp=torch.nn.parallel.DistributedDataParallel(fresh,device_ids=[rank],**configure_quantized_ddp().to_kwargs())
    buffer.seek(0);saved=torch.load(buffer,weights_only=False)
    fresh.load_state_dict(saved['model']);newopt.load_state_dict(saved['optimizer'])
    assert_same(snapshot(fresh),current)
    assert step(newddp,newopt,device,2,rank)==loss
    assert_same(snapshot(fresh),final)
    for name,t in model_storage(fresh):
        ref=t.detach().clone().contiguous();dist.broadcast(ref,0)
        torch.testing.assert_close(t,ref,rtol=0,atol=0)
    print(f'PASS DDP rank={rank} Hadamard+compile+Kahan exact update/resume peak_GB={torch.cuda.max_memory_allocated()/1e9:.3f}',flush=True)
    torch._dynamo.reset()
