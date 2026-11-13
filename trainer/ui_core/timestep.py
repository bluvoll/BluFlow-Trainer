from dataclasses import dataclass
import math
from ..training.flow import get_lin_function, sample_timesteps
from ..training.moss_timesteps import probability_table

SAMPLER_KEYS = (
    "timestep_sample_method",
    "sigmoid_scale",
    "shift",
    "flux_shift",
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


def distribution(cfg, latent_h=64, latent_w=64, bins=40):
    """Exact bin integrals (or sums for the discrete recipe), before curriculum mapping."""
    if cfg.timestep_sample_method == "moss":
        p = probability_table(
            cfg.moss_mu,
            cfg.moss_std,
            cfg.moss_cutoff,
            cfg.moss_uniform_mix,
            cfg.moss_p_full,
            cfg.moss_steps,
        ).tolist()
        masses = [0.0] * bins
        high = 0.0
        for j, mass in enumerate(p, 1):
            # Integer arithmetic avoids moving a grid point across a bin edge by rounding.
            masses[min(j * bins // cfg.moss_steps, bins - 1)] += mass
            if 10 * j >= 9 * cfg.moss_steps:
                high += mass
        median = sample_timesteps(
            cfg, 1, latent_h, latent_w, "cpu", quantile=0.5
        ).item()
    else:
        shift = (
            cfg.shift
            if cfg.shift is not None
            else (
                math.exp(get_lin_function(y1=0.5, y2=1.15)(latent_h * latent_w))
                if cfg.flux_shift
                else 1
            )
        )
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("Shift must be finite and positive")

        def cdf(t):
            if t <= 0 or t >= 1:
                return float(t >= 1)
            # Inverse of the trainer's static shift (Flux shift has sigma=1).
            base = t / (shift + (1 - shift) * t)
            if cfg.timestep_sample_method == "uniform":
                return base
            z = (math.log(base) - math.log1p(-base)) / cfg.sigmoid_scale
            return 0.5 * math.erfc(-z / math.sqrt(2))

        edges = [cdf(j / bins) for j in range(bins + 1)]
        masses = [max(0, b - a) for a, b in zip(edges, edges[1:])]
        median = shift / (1 + shift)
        high = 1 - cdf(0.9)
    return Distribution(tuple(masses), median, high)
