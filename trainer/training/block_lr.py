"""Experimental MOSS learning-rate multipliers, indexed by joint block."""
from dataclasses import dataclass, field
import math


BLOCK_COUNT = 42


def default_ranges():
    return [dict(start=start, end=end, lr_mul=1.0)
            for start, end in ((0, 24), (25, 32), (33, 41))]


@dataclass
class BlockLRConfig:
    enabled: bool = False
    ranges: list[dict] = field(default_factory=default_ranges)
    min_lr_mul: float = 0.1

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('block_lr.enabled must be true or false')
        if not isinstance(self.ranges, list):
            raise ValueError('block_lr.ranges must be an array of block ranges')
        self._validate_multiplier(self.min_lr_mul, 'min_lr_mul')
        covered = set()
        required = {'start', 'end', 'lr_mul'}
        allowed = required | {'scale_by_depth'}
        for row in self.ranges:
            if not isinstance(row, dict) or not required <= row.keys() <= allowed:
                raise ValueError('Each block_lr range needs start, end and lr_mul, with optional scale_by_depth')
            if type(row.get('scale_by_depth', False)) is not bool:
                raise ValueError('block_lr.scale_by_depth must be true or false')
            start, end, multiplier = row['start'], row['end'], row['lr_mul']
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start <= end < BLOCK_COUNT):
                raise ValueError('block_lr ranges must have inclusive integer indices 0 <= start <= end <= 41')
            self._validate_multiplier(multiplier, 'lr_mul')
            indices = set(range(start, end + 1))
            if covered & indices:
                raise ValueError('block_lr ranges must not overlap')
            covered.update(indices)

    @staticmethod
    def _validate_multiplier(value, name):
        try:
            if isinstance(value, (float, int)) and not isinstance(value, bool) and value >= 0 and math.isfinite(value):
                return
        except OverflowError:
            pass
        raise ValueError(f'block_lr.{name} must be finite and nonnegative')

    def expanded(self):
        """Unlisted blocks, and all blocks while disabled, retain multiplier 1."""
        values = [1.0] * BLOCK_COUNT
        if not self.enabled:
            return values
        previous_target = float(self.min_lr_mul)
        for row in sorted(self.ranges, key=lambda row: row['start']):
            start, end, target = row['start'], row['end'], float(row['lr_mul'])
            if row.get('scale_by_depth') and start < end and previous_target != target:
                values[start] = previous_target
                for block in range(start + 1, end):
                    fraction = (block - start) / (end - start)
                    values[block] = previous_target + (target - previous_target) * fraction
                values[end] = target
            else:
                values[start:end + 1] = [target] * (end - start + 1)
            previous_target = target
        return values

    def check_resume(self, saved_multipliers):
        # Old states predate this option and used multiplier 1 everywhere.
        expected = [1.0] * BLOCK_COUNT if saved_multipliers is None else saved_multipliers
        if expected != self.expanded():
            raise ValueError('Full-state resume block LR multipliers changed; use weights-only adapter resume or load the finetuned checkpoint as model_path in a fresh run to start a new LR plan')

    def summary(self):
        values = self.expanded()
        spans = []
        start = 0
        for end in range(BLOCK_COUNT):
            if end == BLOCK_COUNT - 1 or values[end + 1] != values[start]:
                spans.append(f'{start}-{end}: x{values[start]:g}')
                start = end + 1
        return 'block LR ' + ', '.join(spans) + ' (text refiner unchanged)'
