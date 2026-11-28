"""Packed native-resolution batch samplers: legacy fixed-count and MOSS token-budget."""

import random
from .dataset import BucketBatchSampler


def estimate_entry_tokens(entry, max_text_tokens: int = 512) -> int:
    """Estimate image + text tokens for one dataset entry (16px stride + conservative caption bound)."""
    w, h = entry.bucket
    img_tokens = (w // 16) * (h // 16)
    candidates = [getattr(entry, "tags", "") or "", getattr(entry, "nl", "") or ""]
    moss = getattr(entry, "moss", None)
    if isinstance(moss, dict):
        candidates.extend(v for v in moss.values() if isinstance(v, str))
    max_chars = max((len(text) for text in candidates), default=0)
    # ~4 chars/token plus Qwen chat-template overhead, bounded by max_text_tokens.
    text_tokens = min(max(1, int(max_text_tokens)), max(16, (max_chars + 3) // 4 + 16))
    return img_tokens + text_tokens


class NativeResolutionBatchSampler(BucketBatchSampler):
    """Legacy fixed sample-count batches spanning native image resolutions."""

    def _batches(self):
        indices = (self.dataset.epoch_indices(self.epoch) if getattr(self.dataset, 'weighted', False)
                   else list(range(len(self.dataset))))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        n = self._flat
        return [
            indices[i : i + n]
            for i in range(0, len(indices), n)
            if not self.drop_last or len(indices[i : i + n]) == n
        ]

    def __len__(self):
        n = (len(self.dataset.epoch_indices(self.epoch)) if getattr(self.dataset, 'weighted', False)
             else len(self.dataset))
        return n // self._flat if self.drop_last else (n + self._flat - 1) // self._flat


class TokenBudgetBatchSampler(BucketBatchSampler):
    """MOSS paper-style token-budget packing: pack variable counts up to `token_budget` tokens.

    Packs shuffled entries via First-Fit into bins capped at `token_budget` (oversized single
    samples run as singletons so nothing is dropped), then pairs similarly filled packs before
    shuffling pairs so DDP ranks receiving adjacent batches stay balanced in token count and VRAM.
    """

    def __init__(
        self,
        dataset,
        token_budget: int = 8576,
        max_text_tokens: int = 512,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        ddp_group_size: int = 2,
    ):
        super().__init__(dataset, batch_size=1, shuffle=shuffle, drop_last=drop_last, seed=seed)
        self.token_budget = int(token_budget)
        self.max_text_tokens = int(max_text_tokens)
        self.ddp_group_size = max(2, int(ddp_group_size))
        self._entry_tokens = [
            estimate_entry_tokens(e, self.max_text_tokens) for e in dataset.entries
        ]
        self._cached_epoch: int | None = None
        self._cached_batches: list[list[int]] | None = None

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.epoch:
            self._cached_epoch = None
            self._cached_batches = None
        super().set_epoch(epoch)

    def _batches(self) -> list[list[int]]:
        if self._cached_epoch == self.epoch and self._cached_batches is not None:
            return self._cached_batches
        indices = (
            self.dataset.epoch_indices(self.epoch)
            if getattr(self.dataset, "weighted", False)
            else list(range(len(self.dataset)))
        )
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)

        open_packs: list[tuple[list[int], int]] = []
        closed_packs: list[tuple[list[int], int]] = []
        budget = self.token_budget
        for idx in indices:
            cost = self._entry_tokens[idx]
            if cost >= budget:
                closed_packs.append(([idx], cost))
                continue
            placed = False
            for pos, (pack, used) in enumerate(open_packs):
                if used + cost <= budget:
                    pack.append(idx)
                    new_used = used + cost
                    open_packs[pos] = (pack, new_used)
                    placed = True
                    break
            if not placed:
                open_packs.append(([idx], cost))

        all_packs = closed_packs + open_packs
        all_packs.sort(key=lambda item: item[1], reverse=True)
        # Keep only complete DDP-sized groups in the shuffled pool; keep any trailing
        # remainder at the end so it never shifts subsequent rank-matched groups out of phase.
        group = self.ddp_group_size
        full_end = (len(all_packs) // group) * group
        groups = [all_packs[i : i + group] for i in range(0, full_end, group)]
        tail = all_packs[full_end:]
        if self.shuffle:
            rng.shuffle(groups)
        batches = [pack for chunk in groups for pack, _ in chunk] + [pack for pack, _ in tail]
        self._cached_epoch = self.epoch
        self._cached_batches = batches
        return batches

    def __len__(self) -> int:
        return len(self._batches())

    def pack_stats(self) -> tuple[float, float, int]:
        batches = self._batches()
        if not batches:
            return 0.0, 0.0, 0
        total_samples = sum(len(b) for b in batches)
        total_tokens = sum(sum(self._entry_tokens[i] for i in b) for b in batches)
        oversized = sum(1 for c in self._entry_tokens if c > self.token_budget)
        return total_samples / len(batches), total_tokens / len(batches), oversized


def collate_native(batch):
    key = "pixels" if "pixels" in batch[0] else "latents"
    out = {
        key: [item[key].unsqueeze(0) for item in batch],
        "captions": [item["caption"] for item in batch],
        "bucket": [item["bucket"] for item in batch],
        "paths": [item["path"] for item in batch],
    }
    if "mask" in batch[0]:
        out["mask"] = [item["mask"].unsqueeze(0) for item in batch]
    return out
