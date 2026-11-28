"""Global sample weighting and coordinated nonfinite-update decisions."""
import torch
import torch.distributed as dist


def global_sample_scale(local_samples, accumulation_steps, device):
    """Undo Accelerate's accumulation divisor and DDP's rank mean.

    Each backward receives the SUM of weighted losses in that microbatch
    (per-image sums in standard mode, or per-token sums in token-budget packing).
    After reduction, divide by the actual total weight over the whole update.
    This also handles partial accumulation groups and uneven bucket/pack batches.
    """
    if isinstance(local_samples, torch.Tensor):
        count = local_samples.detach().to(device=device, dtype=torch.float64)
        positive_local = False
    else:
        count = torch.tensor(local_samples, device=device, dtype=torch.float64)
        positive_local = local_samples > 0
    world = 1
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(count)
    # Positive host counts guarantee count > 0 without forcing a CPU-GPU synchronization
    # before scaling/clipping kernels are queued.
    if not positive_local and count.item() <= 0:
        raise ValueError('An optimizer update must contain at least one image')
    return (world * accumulation_steps / count).to(torch.float32)


def scale_gradients(parameters, scale):
    grads = [p.grad for p in parameters if p.grad is not None]
    if grads:
        torch._foreach_mul_(grads, scale)


def gradient_norm(parameters, device):
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros((), device=device)
    per_tensor = torch._foreach_norm(grads, 2.0)
    return torch.linalg.vector_norm(torch.stack([n.float() for n in per_tensor]))


def clip_gradients_with_norm(parameters, max_grad_norm, total_norm):
    """Clip gradients using an already-computed global L2 norm to avoid a second pass."""
    if max_grad_norm <= 0:
        return
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    clip_coef = (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)
    torch._foreach_mul_(grads, clip_coef)


def globally_finite(loss_finite, norm):
    bad = (~loss_finite.bool() | ~torch.isfinite(norm)).to(torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    return not bool(bad.item())


def nonfinite_names(named_parameters, limit=8):
    names = []
    for name, p in named_parameters:
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            names.append(name)
            if len(names) == limit:
                break
    return names


def guard_released_gradient(parameter):
    """Installed BEFORE optimizer hooks: stop, never continue a partial update."""
    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
        raise FloatingPointError(
            'Nonfinite gradient during gradient release. Training stopped; earlier parameters '
            'may already have updated, so this partial step cannot be skipped or saved safely.')
