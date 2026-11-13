"""Shared timestep probability laws and Cabal sampling adjustments.

Cabal's shift curve and low-SNR deficit redistribution are adapted from
cabal-SDXL-trainer; see NOTICE. Previewing these laws never consumes RNG.
"""

import bisect
import math
from functools import lru_cache
from itertools import accumulate
from statistics import NormalDist
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .flow import FlowConfig


def cabal_shift(
    latent_h: float, latent_w: float, anchor: float = 6.0, offset: float = 0.0
) -> float:
    """Reference curve, anchored at 1024px (a 64×64 latent grid).

    All current trainer VAEs produce spatial grids downsampled by 16. This is
    the reference's pixel-area/(8² × 2²) sequence length, including non-squares.
    """
    sequence = latent_h * latent_w
    bounded = min(16384.0, max(256.0, sequence))
    if bounded <= 1024:
        raw = 0.33 + (bounded - 256) / (1024 - 256) * (1 - 0.33)
    else:
        raw = 1 + (bounded - 1024) / (16384 - 1024) * (8.5 - 1)
    shift = raw + (anchor - 2.5 + offset) * math.sqrt(sequence / 4096)
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError(
            f"Cabal shift must be positive at {latent_h * 16}×{latent_w * 16}px; "
            f"got {shift:g}. Increase flow.shift or flow.cabal_shift_offset."
        )
    return shift


def effective_shift(cfg: "FlowConfig", latent_h: float, latent_w: float) -> float:
    if cfg.cabal_shift:
        return cabal_shift(latent_h, latent_w, cfg.shift, cfg.cabal_shift_offset)
    if cfg.shift is not None:
        shift = cfg.shift
    elif cfg.flux_shift:
        from .flow import get_lin_function

        shift = math.exp(get_lin_function(y1=0.5, y2=1.15)(latent_h * latent_w))
    else:
        shift = 1.0
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("Shift must be finite and positive")
    return shift


def _sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


class ProbabilityLaw:
    def __init__(self, method, mean, std, shift, moss, lo, hi, mapping):
        self.method, self.mean, self.std, self.shift = method, mean, std, shift
        self.lo, self.hi, self.mapping = lo, hi, mapping
        self.points = None
        if method == "moss":
            from .moss_timesteps import probability_table

            masses = probability_table(*moss).tolist()
            steps = len(masses)
            if mapping == "rescale":
                points = [lo + (hi - lo) * (j + 1) / steps for j in range(steps)]
            else:
                allowed = [
                    ((j + 1) / steps, mass)
                    for j, mass in enumerate(masses)
                    if lo <= (j + 1) / steps <= hi
                ]
                points = [point for point, _ in allowed]
                masses = [mass for _, mass in allowed]
            total = math.fsum(masses)
            if total <= 0:
                raise ValueError("The timestep range contains no MOSS probability mass")
            self.points = tuple(points)
            self.masses = tuple(p / total for p in masses)
            self.cumulative = tuple(accumulate(self.masses))[:-1] + (1.0,)
        else:
            self.start = self._source_cdf(lo) if mapping == "truncate" else 0.0
            self.mass = (
                self._source_cdf(hi) - self.start if mapping == "truncate" else 1.0
            )
            if self.mass <= 0:
                raise ValueError(
                    "The timestep range contains no representable probability mass"
                )

    def _source_cdf(self, t):
        if t <= 0 or t >= 1:
            return float(t >= 1)
        base = t / (self.shift + (1 - self.shift) * t)
        if self.method == "uniform":
            return base
        z = (math.log(base) - math.log1p(-base) - self.mean) / self.std
        return 0.5 * math.erfc(-z / math.sqrt(2))

    def cdf(self, t: float, inclusive: bool = True) -> float:
        if self.points is not None:
            index = (bisect.bisect_right if inclusive else bisect.bisect_left)(
                self.points, t
            )
            return self.cumulative[index - 1] if index else 0.0
        if t <= self.lo or t >= self.hi:
            return float(t >= self.hi)
        if self.mapping == "rescale":
            return self._source_cdf((t - self.lo) / (self.hi - self.lo))
        return min(1.0, max(0.0, (self._source_cdf(t) - self.start) / self.mass))

    def quantile(self, q: float) -> float:
        if self.points is not None:
            return self.points[
                min(len(self.points) - 1, bisect.bisect_left(self.cumulative, q))
            ]
        source_q = self.start + q * self.mass
        if source_q <= 0:
            value = 0.0
        elif source_q >= 1:
            value = 1.0
        elif self.method == "uniform":
            value = source_q
        else:
            value = _sigmoid(self.mean + self.std * NormalDist().inv_cdf(source_q))
        value = value * self.shift / (1 + (self.shift - 1) * value)
        return (
            self.lo + (self.hi - self.lo) * value
            if self.mapping == "rescale"
            else value
        )


class LowSNRLaw:
    """Move donor mass into deficient tail bins, retaining the base draw otherwise."""

    def __init__(self, base):
        self.base = base
        self.points = base.points
        self.lo, self.hi = base.lo, base.hi
        self.body = base.cdf(0.95, inclusive=False)
        if base.points is not None:
            intervals = [(point, point) for point in base.points if point >= 0.95]
            deficits = [
                max(0.0, 1 / len(base.points) - mass)
                for point, mass in zip(base.points, base.masses)
                if point >= 0.95
            ]
        else:
            intervals = [
                (max(base.lo, j / 1000), min(base.hi, (j + 1) / 1000))
                for j in range(950, 1000)
                if max(base.lo, j / 1000) < min(base.hi, (j + 1) / 1000)
            ]
            deficits = [
                max(0.0, (b - a) / (base.hi - base.lo) - (base.cdf(b) - base.cdf(a)))
                for a, b in intervals
            ]
        total = math.fsum(deficits)
        self.move_probability = (
            min(1.0, total / self.body) if total > 1e-12 and self.body > 1e-12 else 0.0
        )
        self.moved = self.body * self.move_probability
        # Drop zero-mass intervals; searchsorted then has no degenerate intervals.
        self.intervals = tuple(
            interval for interval, deficit in zip(intervals, deficits) if deficit > 0
        )
        weights = (
            [deficit / total for deficit in deficits if deficit > 0] if total else []
        )
        self.replacement_cdf = (
            tuple(accumulate(weights))[:-1] + (1.0,) if weights else ()
        )
        self._device_tables = {}

    def _replacement_cdf(self, t, inclusive):
        result = 0.0
        previous = 0.0
        for (a, b), cumulative in zip(self.intervals, self.replacement_cdf):
            weight = cumulative - previous
            if a == b:
                fraction = float(t >= a if inclusive else t > a)
            else:
                fraction = min(1.0, max(0.0, (t - a) / (b - a)))
            result += weight * fraction
            previous = cumulative
        return result

    def cdf(self, t: float, inclusive: bool = True) -> float:
        original = self.base.cdf(t, inclusive)
        if not self.move_probability:
            return original
        return min(
            1.0,
            max(
                0.0,
                original
                - self.move_probability * min(original, self.body)
                + self.moved * self._replacement_cdf(t, inclusive),
            ),
        )

    def quantile(self, q: float) -> float:
        if not self.move_probability:
            return self.base.quantile(q)
        if self.points is not None:
            left, right = 0, len(self.points) - 1
            while left < right:
                middle = (left + right) // 2
                if self.cdf(self.points[middle]) >= q:
                    right = middle
                else:
                    left = middle + 1
            return self.points[left]
        left, right = self.lo, self.hi
        for _ in range(55):
            middle = (left + right) / 2
            if self.cdf(middle) >= q:
                right = middle
            else:
                left = middle
        return (left + right) / 2

    def adjust(
        self, values: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        if not self.move_probability:
            return values
        device = generator.device if generator is not None else "cpu"
        choose = torch.rand(values.shape, device=device, generator=generator).to(
            values.device
        )
        u = (
            torch.rand(values.shape, device=device, generator=generator)
            .to(values.device)
            .double()
        )
        key = str(values.device)
        if key not in self._device_tables:
            self._device_tables[key] = (
                torch.tensor(
                    self.replacement_cdf, dtype=torch.float64, device=values.device
                ),
                torch.tensor(self.intervals, dtype=torch.float64, device=values.device),
            )
        cumulative, intervals = self._device_tables[key]
        index = torch.searchsorted(cumulative, u, right=True).clamp_max(
            len(cumulative) - 1
        )
        before = torch.cat((cumulative.new_zeros(1), cumulative[:-1]))[index]
        fraction = (u - before) / (cumulative[index] - before)
        replacements = intervals[index, 0] + fraction * (
            intervals[index, 1] - intervals[index, 0]
        )
        return torch.where(
            (values < 0.95) & (choose < self.move_probability),
            replacements.to(values.dtype),
            values,
        )


_base_law = lru_cache(maxsize=64)(ProbabilityLaw)
_corrected_law = lru_cache(maxsize=64)(LowSNRLaw)


def probability_law(
    cfg: "FlowConfig",
    latent_h: float = 64,
    latent_w: float = 64,
    t_range: tuple[float, float] | None = None,
) -> ProbabilityLaw | LowSNRLaw:
    lo, hi = t_range if t_range is not None else (0.0, 1.0)
    if not 0 <= lo < hi <= 1:
        raise ValueError("Timestep range must satisfy 0 <= low < high <= 1")
    law = _base_law(
        cfg.timestep_sample_method,
        cfg.logit_mean,
        cfg.sigmoid_scale,
        effective_shift(cfg, latent_h, latent_w),
        (
            cfg.moss_mu,
            cfg.moss_std,
            cfg.moss_cutoff,
            cfg.moss_uniform_mix,
            cfg.moss_p_full,
            cfg.moss_steps,
        ),
        lo,
        hi,
        cfg.phase_mapping,
    )
    return _corrected_law(law) if cfg.normal_low_snr else law
