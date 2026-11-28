"""Frozen MOSS INT8 projections with a compile-safe input-gradient path.

SDNQ supplies quantization and GEMM kernels. Custom operators keep its
Python/tensor-subclass machinery out of compiled transformer blocks. Only
adapter parameters are trainable; these operators do not compute base gradients.
"""
from __future__ import annotations

from functools import lru_cache
import os
import torch
from torch import nn
from sdnq.dequantizer import dequantize_symmetric_compiled
from sdnq.kernel_wrappers import int_scaled_mm_func
from sdnq.quant_utils import get_hadamard, rotate_hadamard_compiled
from sdnq.training import SDNQTensor
from sdnq.training.layers.linear.linear_int8.linear_int8 import get_int8_matmul_inputs
from sdnq.training.layers.linear.linear_int8.linear_int8_dynamic import get_int8_matmul_dynamic_inputs
from .int8_calibration import launch_scaled_mm, select_tile


def _dynamic_helper(fn):
    # SDNQ defaults to static internal helper compilation. Do not change process
    # environment or monkeypatch its globals: text encoding may already import it.
    return torch.compile(getattr(fn, "_torchdynamo_orig_callable", fn), fullgraph=True, dynamic=True)


_forward_inputs = _dynamic_helper(get_int8_matmul_inputs)
_backward_inputs = _dynamic_helper(get_int8_matmul_dynamic_inputs)
_rotate = _dynamic_helper(rotate_hadamard_compiled)


@lru_cache(maxsize=None)
def _is_ada(device):
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (8, 9)


def uses_fixed_ada_tile(device):
    """Legacy 0.2.4 workaround; newer releases provide hardware-aware tuning."""
    from sdnq import kernel_wrappers
    from sdnq.kernels import triton_scaled_mm as kernel
    return (hasattr(kernel, 'min_block_size')
            and kernel_wrappers.use_triton_scaled_mm and _is_ada(device)
            and not any(key.startswith("SDNQ_TRITON_MM_") for key in os.environ))


def _ada_scaled_mm(x, weight, x_scale, scale, dtype):
    # SDNQ 0.2.4 tunes 18 candidates for every new ceil(tokens/256) key.
    # Packed DDP gives each rank different keys: ~6s steps became ~60s while
    # the other rank waited. All measured MOSS projection shapes on RTX 4090
    # selected this tile. Reuse SDNQ's exact INT32-accumulating kernel without
    # its tuner (or its 256 MiB benchmark buffer). New shapes still JIT compile.
    from .int8_calibration import launch_scaled_mm
    tile = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128,
                GROUP_SIZE_M=8, num_warps=4, num_stages=2)
    return launch_scaled_mm(x, weight, x_scale, scale, dtype, tile)



def _gemm(inputs):
    x, weight, x_scale, scale, bias, after, dtype, shape = inputs
    assert after is None
    if bias is None:
        tile = select_tile(x.device, x.shape[0], weight.shape[1], x.shape[1], weight.is_contiguous())
        if tile is not None:
            return launch_scaled_mm(x, weight, x_scale, scale, dtype, tile).view(shape)
    if bias is None and uses_fixed_ada_tile(x.device):
        return _ada_scaled_mm(x, weight, x_scale, scale, dtype).view(shape)
    return int_scaled_mm_func(x, weight, x_scale, scale, bias=bias, out_dtype=dtype).view(shape)


@torch.library.custom_op("bluflow::frozen_int8_forward", mutates_args=())
def _forward(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
             rotation: torch.Tensor | None) -> torch.Tensor:
    return _gemm(_forward_inputs(x, weight, scale, hadamard=rotation, do_transpose=True))


@_forward.register_fake
def _forward_fake(x, weight, scale, rotation):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@torch.library.custom_op("bluflow::frozen_int8_backward", mutates_args=())
def _backward(grad: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
              rotation: torch.Tensor | None) -> torch.Tensor:
    shape = (*grad.shape[:-1], weight.shape[1])
    decoded = dequantize_symmetric_compiled(weight, scale)
    result = _gemm(_backward_inputs(grad.flatten(0, -2), decoded, hadamard=rotation,
                                   output_shape=shape, do_input_reshape=False))
    return _rotate(result, hadamard=rotation) if rotation is not None else result


@_backward.register_fake
def _backward_fake(grad, weight, scale, rotation):
    return grad.new_empty((*grad.shape[:-1], weight.shape[1]))


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[1:])


def _autograd_backward(ctx, grad):
    return _backward(grad, *ctx.saved_tensors), None, None, None


_forward.register_autograd(_autograd_backward, setup_context=_setup_context)


@torch.library.custom_op("bluflow::frozen_int8_qkv_forward", mutates_args=())
def _forward_qkv(
    x: torch.Tensor,
    wq: torch.Tensor,
    sq: torch.Tensor,
    wk: torch.Tensor,
    sk: torch.Tensor,
    wv: torch.Tensor,
    sv: torch.Tensor,
    rotation: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Rotate and dynamically INT8-quantize `x` once, then reuse `(x_q, x_scale)` across Q/K/V.
    x_q, wq_t, x_scale, sq_t, bias, after, dtype, shape_q = _forward_inputs(
        x, wq, sq, hadamard=rotation, do_transpose=True
    )
    prefix = x.shape[:-1]
    contiguous = wq_t.is_contiguous()
    wk_t = wk.t().contiguous() if contiguous else wk.t()
    wv_t = wv.t().contiguous() if contiguous else wv.t()
    out_q = _gemm((x_q, wq_t, x_scale, sq_t, bias, after, dtype, shape_q))
    out_k = _gemm((x_q, wk_t, x_scale, sk.t(), None, None, dtype, (*prefix, wk.shape[0])))
    out_v = _gemm((x_q, wv_t, x_scale, sv.t(), None, None, dtype, (*prefix, wv.shape[0])))
    return out_q, out_k, out_v


@_forward_qkv.register_fake
def _forward_qkv_fake(x, wq, sq, wk, sk, wv, sv, rotation):
    prefix = x.shape[:-1]
    return (
        x.new_empty((*prefix, wq.shape[0])),
        x.new_empty((*prefix, wk.shape[0])),
        x.new_empty((*prefix, wv.shape[0])),
    )


def _single_pre_rot_backward(
    grad: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, rotation: torch.Tensor | None
) -> torch.Tensor:
    shape = (*grad.shape[:-1], weight.shape[1])
    decoded = dequantize_symmetric_compiled(weight, scale)
    return _gemm(
        _backward_inputs(
            grad.flatten(0, -2), decoded, hadamard=rotation, output_shape=shape, do_input_reshape=False
        )
    )


@torch.library.custom_op("bluflow::frozen_int8_qkv_backward", mutates_args=())
def _backward_qkv(
    gq: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    wq: torch.Tensor,
    sq: torch.Tensor,
    wk: torch.Tensor,
    sk: torch.Tensor,
    wv: torch.Tensor,
    sv: torch.Tensor,
    rotation: torch.Tensor | None,
) -> torch.Tensor:
    dx = (
        _single_pre_rot_backward(gq, wq, sq, rotation)
        + _single_pre_rot_backward(gk, wk, sk, rotation)
        + _single_pre_rot_backward(gv, wv, sv, rotation)
    )
    return _rotate(dx, hadamard=rotation) if rotation is not None else dx


@_backward_qkv.register_fake
def _backward_qkv_fake(gq, gk, gv, wq, sq, wk, sk, wv, sv, rotation):
    return gq.new_empty((*gq.shape[:-1], wq.shape[1]))


def _setup_qkv_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[1:])


def _autograd_qkv_backward(ctx, gq, gk, gv):
    wq, sq, wk, sk, wv, sv, rotation = ctx.saved_tensors
    if not all(g is not None for g in (gq, gk, gv)):
        ref = next((g for g in (gq, gk, gv) if g is not None), None)
        if ref is None:
            return (None, None, None, None, None, None, None, None)
        prefix = ref.shape[:-1]
        if gq is None:
            gq = ref.new_zeros((*prefix, wq.shape[0]))
        if gk is None:
            gk = ref.new_zeros((*prefix, wk.shape[0]))
        if gv is None:
            gv = ref.new_zeros((*prefix, wv.shape[0]))
    dx = _backward_qkv(gq, gk, gv, wq, sq, wk, sk, wv, sv, rotation)
    return dx, None, None, None, None, None, None, None


_forward_qkv.register_autograd(_autograd_qkv_backward, setup_context=_setup_qkv_context)


def _extract_qkv_part(mod):
    if isinstance(mod, FrozenINT8Linear):
        return (mod, None) if mod.bias is None else (None, None)
    base = getattr(mod, "base_layer", None)
    if (
        isinstance(base, FrozenINT8Linear)
        and base.bias is None
        and not getattr(mod, "merged", False)
        and not getattr(mod, "disable_adapters", False)
        and not getattr(mod, "lora_variant", {})
        and hasattr(mod, "lora_A")
        and hasattr(mod, "lora_B")
    ):
        return base, mod
    return None, None


def _apply_peft_delta(base_out: torch.Tensor, x: torch.Tensor, mod) -> torch.Tensor:
    if mod is None or getattr(mod, "disable_adapters", False):
        return base_out
    res = base_out
    out_dtype = base_out.dtype
    for active_adapter in mod.active_adapters:
        if active_adapter not in mod.lora_A:
            continue
        lora_A = mod.lora_A[active_adapter]
        x_in = mod._cast_input_dtype(x, lora_A.weight.dtype)
        res = (
            res
            + mod.lora_B[active_adapter](lora_A(mod.lora_dropout[active_adapter](x_in)))
            * mod.scaling[active_adapter]
        )
    return res.to(out_dtype)


def apply_shared_frozen_qkv(mod_q, mod_k, mod_v, x: torch.Tensor):
    """Compute Q, K, V projections sharing a single FrozenINT8Linear input prep/backward rotation.

    Returns `(q, k, v)` when all three layers use compatible `FrozenINT8Linear` base layers
    (directly or wrapped by vanilla PEFT LoRA), else `None` so callers can fall back.
    """
    bq, pq = _extract_qkv_part(mod_q)
    bk, pk = _extract_qkv_part(mod_k)
    bv, pv = _extract_qkv_part(mod_v)
    if bq is None or bk is None or bv is None:
        return None
    rq, rk, rv = bq._sdnq_rotation, bk._sdnq_rotation, bv._sdnq_rotation
    if (rq is None) != (rk is None) or (rq is None) != (rv is None):
        return None
    if rq is not None and (rq.shape != rk.shape or rq.shape != rv.shape):
        return None
    base_q, base_k, base_v = _forward_qkv(
        x,
        bq.weight,
        bq._sdnq_scale,
        bk.weight,
        bk._sdnq_scale,
        bv.weight,
        bv._sdnq_scale,
        rq,
    )
    return (
        _apply_peft_delta(base_q, x, pq),
        _apply_peft_delta(base_k, x, pk),
        _apply_peft_delta(base_v, x, pv),
    )


class FrozenINT8Linear(nn.Linear):
    """Plain INT8 storage plus persistent scales, suitable for moves/resume/DDP."""

    _trainer_sdnq_matmul = True

    def __init__(self, source):
        w = source.weight
        if (source.bias is not None or w.requires_grad or w.zero_point is not None
                or w.svd_up is not None or w.svd_down is not None
                or w.weight.dtype != torch.int8 or tuple(w.weight.shape) != tuple(w.shape)):
            raise ValueError("Frozen INT8 matmul requires bias-free symmetric row-wise frozen INT8 weights")
        super().__init__(source.in_features, source.out_features, bias=False, device="meta")
        self.weight = nn.Parameter(w.weight, requires_grad=False)
        self.register_buffer("_sdnq_scale", w.scale.float())
        rotation = (get_hadamard(256, dtype=torch.bfloat16, device=w.device)
                    if w.sdnq_dequantizer.use_hadamard else None)
        self.register_buffer("_sdnq_rotation", rotation)
        self.train(source.training)

    def _apply(self, fn, recurse=True):
        # Module.bfloat16()/to(dtype=...) must never round the FP32 scales.
        # The INT8 weight follows device moves but retains its integer dtype.
        scale = self._buffers.pop("_sdnq_scale")
        try:
            super()._apply(fn, recurse=recurse)
        finally:
            self._buffers["_sdnq_scale"] = scale.to(device=self.weight.device)
        return self

    def forward(self, x):
        return _forward(x, self.weight, self._sdnq_scale, self._sdnq_rotation)


def install_frozen_matmul(model):
    targets = [(name, layer) for name, layer in model.named_modules()
               if isinstance(getattr(layer, "weight", None), SDNQTensor)]
    if not targets:
        raise ValueError("Frozen INT8 matmul selected no eligible projections")
    for name, layer in targets:
        replacement = FrozenINT8Linear(layer)
        if not name:
            return replacement
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, child, replacement)
    return model
