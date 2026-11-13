"""Rank-sharded, full-size batches from a single tier's deterministic sample stream."""
import hashlib
import json
import random
import math
from dataclasses import asdict
from pathlib import Path
from .dataset import BucketBatchSampler


class StagedBatchSampler(BucketBatchSampler):
    def __init__(self, dataset, batch_size, schedule, steps, *, seed=0, rank=0, world=1, accumulation=1):
        super().__init__(dataset, batch_size, seed=seed)
        self.schedule = schedule
        self.budgets = schedule.budgets(steps)
        self.rank, self.world, self.accumulation = rank, world, accumulation
        self.indices = {s.resolution: [i for i, e in enumerate(dataset.entries) if e.resolution == s.resolution]
                        for s in schedule.stages}
        self.epoch_mode = schedule.unit == 'epochs'
        self.epoch_budgets = list(self.budgets) if self.epoch_mode else None
        if self.epoch_mode:
            self.epoch_batches = [math.ceil(len(self._indices(0, s.resolution)) /
                                           (self._by_tier[s.resolution] * world)) for s in schedule.stages]
            self.budgets = [n * math.ceil(b / accumulation) for n, b in zip(self.epoch_budgets, self.epoch_batches)]
            if any(b == 0 for b in self.epoch_batches):
                raise ValueError('Every resolution stage needs at least one eligible sample')
        if any(not indices for indices in self.indices.values()):
            raise ValueError('Every resolution stage needs at least one eligible sample')
        self.stage = self.cursor = self.count = 0
        files = {}
        entries = []
        for e in dataset.entries:
            path = str(Path(e.latent_path).resolve())
            if path not in files:
                st = Path(path).stat()
                files[path] = (st.st_size, st.st_mtime_ns)
            entries.append((path, e.bucket, e.resolution, e.tags, e.nl))
        payload = dict(
            entries=entries, files=files, budgets=self.budgets,
            resolutions=[s.resolution for s in schedule.stages], batch_size=batch_size,
            world=world, accumulation=accumulation, seed=seed,
        )
        if self.epoch_mode or getattr(dataset, 'weighted', False):
            payload.update(version=2, unit=schedule.unit, epochs=self.epoch_budgets,
                           weighting=asdict(dataset.config)
                           if getattr(dataset, 'weighted', False) else None)
        self.identity = hashlib.sha256(json.dumps(payload, sort_keys=True, default=sorted).encode()).hexdigest()

    def _indices(self, epoch, resolution):
        return (self.dataset.epoch_indices(epoch, resolution) if getattr(self.dataset, 'weighted', False)
                else self.indices[resolution])

    def _order(self, epoch, stage):
        indices = self._indices(epoch, self.schedule.stages[stage].resolution)
        visits, order = {}, []
        for i in indices:
            occurrence = visits.get(i, 0)
            visits[i] = occurrence + 1
            sample = (i, epoch + 1000003 * stage, None, None)
            order.append(sample + (occurrence,) if getattr(self.dataset, 'weighted', False) else sample)
        random.Random(self.seed + 1000003 * stage + 9176 * epoch).shuffle(order)
        return order

    def configure_epoch(self, stage, epoch, cursor=0):
        self.stage, self.epoch, self.cursor = stage, epoch, cursor
        self.count = self.epoch_batches[stage] - cursor

    def configure(self, stage, cursor, updates):
        self.stage, self.cursor = stage, cursor
        self.count = updates * self.accumulation

    def __len__(self):
        return self.count

    def __iter__(self):
        # Snapshot state before prefetch begins. The consumed cursor is maintained
        # by the training loop, never inferred from how far workers prefetched.
        stage, cursor, count = self.stage, self.cursor, self.count
        if self.epoch_mode:
            order = self._order(self.epoch, stage)
            bs = self._by_tier[self.schedule.stages[stage].resolution]
            for micro in range(cursor, cursor + count):
                first = (micro * self.world + self.rank) * bs
                # Match ordinary epoch loaders: keep the single-rank partial batch;
                # distributed even batches wrap only the final global microbatch.
                end = min(first + bs, len(order)) if self.world == 1 else first + bs
                yield [order[pos % len(order)] for pos in range(first, end)]
            return
        resolution = self.schedule.stages[stage].resolution
        indices = self._indices(0, resolution)
        bs = self._by_tier[resolution]
        cycle_cached, order = -1, None
        for micro in range(cursor, cursor + count):
            first = (micro * self.world + self.rank) * bs
            batch = []
            for pos in range(first, first + bs):
                cycle, offset = divmod(pos, len(indices))
                if cycle != cycle_cached:
                    order = self._order(cycle, stage)
                    cycle_cached = cycle
                # Also give each stage a distinct deterministic caption stream.
                batch.append(order[offset])
            yield batch
