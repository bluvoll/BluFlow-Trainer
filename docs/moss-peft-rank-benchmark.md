# MOSS PEFT rank benchmark

Measured on 2026-11-16 using physical GPU 1: RTX 4090, observed core clock
2400 MHz and memory clock 10801 MHz.

## Setup

Full MOSS base model; PEFT LoRA on joint attention/MLP targets, excluding text
refiners. Batch 4 at 1024×1024, accumulation 1, fixed synthetic latent inputs and
128 text tokens per sample. Text encoder and VAE were not loaded.

BF16 adapters, alpha/rank = 0.5, Balanced SDNQ INT8/Hadamard, packed varlen
attention, regional dynamic compilation and all-block gradient checkpointing
with the default boundary outside compilation. CAME at LR 1e-4 with RMS clipping;
no Kahan, optimizer-state quantization, or offloading. Expandable allocator
segments enabled; SDNQ kernel selection without a user calibration override.

Versions: PyTorch 2.10.0+cu128, PEFT 0.20.0, SDNQ 0.2.9, Triton 3.6.0.
Each rank ran in a fresh process: four warmup updates followed by six measured
updates. All completed with finite losses; compiled graph counts stayed constant
throughout each measured window.

## Results

VRAM is peak PyTorch allocated/reserved memory **after warmup**, in GiB. Reserved
memory includes allocator caching; neither number includes all CUDA context or
external allocations. Time is the median measured forward/backward/optimizer step.

| Rank | Trainable parameters | Allocated / reserved GiB | Seconds/step |
| ---: | ---: | ---: | ---: |
| 16 | 37.2M | 13.20 / 13.76 | 3.714 |
| 32 | 74.3M | 13.34 / 13.96 | 3.706 |
| 64 | 148.6M | 13.62 / 14.28 | 3.721 |
| 128 | 297.3M | 14.19 / 15.03 | 3.750 |
| 256 | 594.5M | 15.32 / 16.06 | 3.931 |

Rank 16 → 64 quadrupled adapter parameters while adding only **0.42 GiB (3.2%)**
to peak allocated VRAM, with essentially unchanged step time. Rank 16 → 256
added **2.12 GiB (16.1%)** and approximately **5.8%** to median step time.

Adapter weights alone grew from 71 MiB at rank 16 to 284 MiB at rank 64 and
1134 MiB at rank 256. Gradients grew proportionally too: the extra rank was
present and participating in training. The frozen model and mostly rank-independent
transformer activations dominate the overall footprint, while BF16 adapters and
CAME keep rank-dependent storage relatively compact. This combined benchmark
does not isolate the savings attributable to each optimization.

## What higher rank enables

For a target matrix, LoRA learns an update of the form `scale × B × A`, whose rank
is at most the configured adapter rank. Rank 64 can therefore represent more
independent update directions than rank 16: it has more capacity to modify the
base weights. It does not automatically increase the update's magnitude or the
inference multiplier. This benchmark measures resource cost, not whether the
additional capacity improves a particular dataset's visual results.
