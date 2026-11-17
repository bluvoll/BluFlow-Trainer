"""PEFT LoKr factorization, training, and external dense-factor scaling."""
import copy
import pytest
import torch
from safetensors.torch import load_file
from peft import LoKrConfig
from peft.tuners.lokr.layer import Linear

from test_moss_image import tiny, inputs
from trainer.training.params import AdapterConfig,apply_adapter
from trainer.modeling.export import export_checkpoint
from trainer.training.adapter_loading import load_adapter_weights
from trainer.training.config import Config
from trainer.data.dataset import DatasetConfig


@pytest.mark.parametrize('rank,dense,expected',[(32,False,32*(3072+16384)+1),(8192,True,3072*16384+1)])
def test_factor_one_does_not_alone_mean_full_rank(rank,dense,expected):
    with torch.device('meta'):
        layer=Linear(torch.nn.Linear(3072,16384,bias=False),'default',r=rank,alpha=rank,
                     config=LoKrConfig(r=rank,alpha=rank,decompose_factor=1))
    assert ('default' in layer.lokr_w2)==dense
    assert layer.lokr_w1['default'].shape==(1,1)
    assert sum(p.numel() for n,p in layer.named_parameters() if 'lokr_' in n)==expected


@pytest.mark.parametrize('factor,rank',[(1,2),(1,100),(4,100),(-1,2)])
@pytest.mark.parametrize('scale',[.5,.85])
def test_lokr_training_export_matches_inference_and_resume(tmp_path,factor,rank,scale):
    torch.manual_seed(123)
    model=tiny();fresh=copy.deepcopy(model)
    cfg=Config(dataset=DatasetConfig(path=str(tmp_path)))
    cfg.train.model_family='moss_image'
    cfg.adapter=AdapterConfig(kind='lokr',rank=rank,alpha=rank*scale,lokr_factor=factor,moss_train_text_refiner=False)
    apply_adapter(model,cfg.adapter);apply_adapter(fresh,cfg.adapter)
    xs,t,text,mask=inputs()
    original=model(xs,t,(text,mask))[0]
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    for _ in range(3):
        loss=sum(x.square().mean() for x in model(xs,t,(text,mask))[0]);loss.backward();opt.step();opt.zero_grad()
    updated=model(xs,t,(text,mask))[0]
    assert any(not torch.equal(a,b) for a,b in zip(original,updated))
    for n,p in model.named_parameters():
        if n in frozen:torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
    export_checkpoint(model,tmp_path,'lokr',cfg,torch.float32,3)
    file=tmp_path/'lokr.safetensors';state=load_file(file)
    for name,module in model.named_modules():
        if not hasattr(module,'lokr_w1'):continue
        prefix='diffusion_model.'+name
        # Reconstruct exactly as Comfy/Forge do, including their dense alpha rule.
        dim=None
        if prefix+'.lokr_w1' in state:w1=state[prefix+'.lokr_w1']
        else:
            a,b=state[prefix+'.lokr_w1_a'],state[prefix+'.lokr_w1_b'];w1=a@b;dim=b.shape[0]
        if prefix+'.lokr_w2' in state:w2=state[prefix+'.lokr_w2']
        else:
            a,b=state[prefix+'.lokr_w2_a'],state[prefix+'.lokr_w2_b'];w2=a@b;dim=b.shape[0]
        scale=state[prefix+'.alpha']/dim if dim is not None else 1.
        actual=torch.kron(w1,w2)*scale
        torch.testing.assert_close(actual,module.get_delta_weight('default'))
    load_adapter_weights(fresh,file)
    for a,b in zip(updated,fresh(xs,t,(text,mask))[0]):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
    if rank==100:
        # Dense tensor shapes stay identical after a rank change; alpha/r does not.
        wrong=tiny();changed=copy.deepcopy(cfg.adapter);changed.rank=200;apply_adapter(wrong,changed)
        with pytest.raises(ValueError,match='rank differs'):load_adapter_weights(wrong,file)


@pytest.mark.parametrize('initialization',['peft','lycoris'])
def test_factor_one_initialization_controls_first_update(initialization):
    cfg=AdapterConfig(kind='lokr',rank=32,alpha=32,lokr_factor=1,lokr_init=initialization)
    layer=Linear(torch.nn.Linear(16,16,bias=False),'default',r=32,alpha=32,config=cfg.build())
    assert torch.count_nonzero(layer.get_delta_weight('default'))==0
    layer(torch.randn(8,16)).square().mean().backward()
    w1_grad=layer.lokr_w1['default'].grad.abs().sum().item()
    w2_grad=layer.lokr_w2['default'].grad.abs().sum().item()
    assert (w1_grad>0)==(initialization=='peft')
    assert (w2_grad>0)==(initialization=='lycoris')


def test_lokr_initialization_shared_gui_roundtrip(tmp_path):
    from trainer.gui import bridge
    from trainer.ui_core.config import evaluate
    from trainer.training.config import load_config
    flat=bridge.defaults()|{'dataset.path':str(tmp_path),'train.model_family':'moss_image',
        'adapter.kind':'lokr','adapter.lokr_factor':1,'adapter.rank':10000,
        'adapter.alpha':10000.,'adapter.lokr_init':'lycoris'}
    assert evaluate(flat)['valid']
    file=tmp_path/'lokr.toml';bridge.write_toml(file,flat)
    assert load_config(file).adapter.build().init_weights=='lycoris'


@pytest.mark.skipif(__import__('os').environ.get('BLUFLOW_GPU_TESTS')!='1',reason='GPU opt-in')
def test_peft_lokr_bf16_compiled_frozen_storage():
    from trainer.modeling.moss_image import MOSSImage,MOSSParams
    from trainer.training.quant import QuantConfig,quantize_module
    from trainer.training.params import build_adapter_param_groups,ComponentLRs
    from trainer.training.optim import build_optimizer
    from trainer.training.config import OptimizerConfig
    from trainer.training.distributed import model_storage
    device=torch.device('cuda');torch.manual_seed(321)
    model=MOSSImage(MOSSParams(dim=256,ffn_dim=512,num_heads=4,num_layers=6,text_dim=64,latent_channels=8)).to(device,torch.bfloat16)
    quantize_module(model,QuantConfig(mode='frozen',weights_dtype='uint8',use_quantized_matmul=False),device,torch.bfloat16,False)
    apply_adapter(model,AdapterConfig(kind='lokr',rank=64,alpha=64,lokr_factor=4,lokr_init='lycoris',dtype='bfloat16',moss_train_text_refiner=False))
    model.configure_execution(True,compile_mode='default',attention_backend='torch_varlen')
    groups=build_adapter_param_groups(model,ComponentLRs(),1e-4)
    opt=build_optimizer(groups.groups,OptimizerConfig(kind='came',lr=1e-4))
    before={n:t.clone() for n,t in model_storage(model) if 'lokr_' not in n}
    adapters={n:p.clone() for n,p in model.named_parameters() if p.requires_grad}
    from test_sdnq_finetune_matmul import step
    for i in range(3):step(model,opt,device,i)
    for n,t in model_storage(model):
        if n in before:torch.testing.assert_close(t,before[n],rtol=0,atol=0)
    assert any(not torch.equal(adapters[n],p) for n,p in model.named_parameters() if n in adapters)
    assert all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
    print('PASS PEFT LoKr BF16 compiled UINT8-storage base unchanged; adapter updates finite',flush=True)
    torch._dynamo.reset()
