"""Deterministic preview integrals of the training sampler's probability laws."""

from dataclasses import dataclass, asdict
from ..training.timestep_distribution import effective_shift, probability_law

SAMPLER_KEYS = (
    "timestep_sample_method",
    "sigmoid_scale",
    "logit_mean",
    "normal_low_snr",
    "shift",
    "flux_shift",
    "cabal_shift",
    "cabal_shift_offset",
    "moss_mu",
    "moss_std",
    "moss_cutoff",
    "moss_uniform_mix",
    "moss_p_full",
    "moss_steps",
)


@dataclass(frozen=True)
class Distribution:
    probabilities: tuple[float, ...]
    median: float
    high_noise: float
    shift: float


def distribution(cfg, latent_h=64, latent_w=64, bins=40):
    """Bin integrals (sums for MOSS), before curriculum range mapping."""
    if type(bins) is not int or bins < 1:
        raise ValueError("Preview bins must be a positive integer")
    law = probability_law(cfg, latent_h, latent_w)
    edges = [law.cdf(j / bins, inclusive=False) for j in range(bins)] + [1.0]
    masses = tuple(max(0.0, b - a) for a, b in zip(edges, edges[1:]))
    return Distribution(
        masses,
        law.quantile(0.5),
        1 - law.cdf(0.9, inclusive=False),
        effective_shift(cfg, latent_h, latent_w),
    )


def resolution_distributions(cfg, flat):
    if not (cfg.flux_shift or cfg.cabal_shift):
        return []
    if flat.get("resolution_schedule.enabled"):
        resolutions = [
            stage["resolution"] for stage in flat.get("resolution_schedule.stages", [])
        ]
    else:
        resolutions = flat.get("dataset.resolutions") or [
            flat.get("dataset.resolution", 1024)
        ]
    if not resolutions or any(type(r) is not int or r <= 0 for r in resolutions):
        raise ValueError(
            "Enter positive resolution tiers to preview the timestep distributions"
        )
    return [
        dict(resolution=r, **asdict(distribution(cfg, r / 16, r / 16, bins=100)))
        for r in sorted(set(resolutions))
    ]
