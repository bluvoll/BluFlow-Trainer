"""Caption construction: tag shuffling, dropout, and tags/NL mixing.

Ported from diffusion-pipe (models/trainer.py:72-520), which is the best-developed part of that
trainer. Both target datasets use the same layout: `<stem>.txt` holds comma-separated tags and
`<stem>_nl.txt` an optional natural-language description.

Live augmentation requires encoding each caption. Persistent caption-variation caching instead
precomputes a finite sequence of augmented captions, preserving duplicate slots while sharing
exact embedding matches. Latent caching remains independent of either text-encoding mode.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

CaptionMode = str
LEGACY_VARIANTS = ("tags", "nl", "tags_nl", "nl_tags")
# Keep medium and layout adjacent so substituting their weights draws matched slots.
MOSS_FALLBACK = {
    "tags_moss_long": "tags_nl", "moss_long_tags": "nl_tags",
    "moss_short": "nl", "moss_long": "nl", "moss_medium": "nl", "layout": "nl", "layout_long": "nl",
}
CAPTION_VARIANTS = (*LEGACY_VARIANTS, *MOSS_FALLBACK)
JSON_MIXES = ("mixed_json", "mixed_json_layout")
# Consume complete quoted inscriptions before looking for sentence boundaries.
_QUOTES_OR_BOUNDARIES = re.compile(r'"[^"\n]*"|“[^”\n]*”|(?<=[.!?])\s+|(?<=[.!?]["”’])\s+')
_JSON_WEIGHTS = dict(tags=30, tags_moss_long=20, moss_long_tags=20, moss_short=5, moss_long=5)
MIX_DEFAULTS = {
    "mixed": dict(tags=50, nl=10, tags_nl=20, nl_tags=20),
    "mixed_json": {**_JSON_WEIGHTS, "moss_medium": 20},
    "mixed_json_layout": {**_JSON_WEIGHTS, "moss_medium": 0, "layout": 20, "layout_long": 0},
    # Only exposed when opening an older recipe combining _nl.txt and MOSS variants.
    "mixed_custom": dict.fromkeys(CAPTION_VARIANTS, 0),
}


def compatible_mix(mode, weights):
    if mode != "mixed" or not weights or not any(weights.get(k, 0) for k in MOSS_FALLBACK):
        return mode
    if any(weights.get(k, 0) for k in LEGACY_VARIANTS if k != "tags"):
        return "mixed_custom"
    return "mixed_json_layout" if weights.get("layout") or weights.get("layout_long") else "mixed_json"


@dataclass
class CaptionConfig:
    caption_mode: CaptionMode = "tags"
    # Percentages; normalised automatically, need not sum to 100.
    mixed_weights: dict[str, float] | None = None

    shuffle_tags: bool = False
    tag_delimiter: str = ", "
    shuffle_keep_first_n: int = 0     # keep N leading tags in place (trigger words)
    tag_dropout_percent: float = 0.0  # fraction of tags dropped per sample
    min_tags_kept: int = 3            # never drop below this many
    protected_tags: set[str] = field(default_factory=set)

    caption_dropout_percent: float = 0.0  # fraction of samples trained unconditional

    nl_shuffle_sentences: bool = False
    nl_keep_first_sentence: bool = False

    def __post_init__(self):
        if self.caption_mode not in (*CAPTION_VARIANTS, *MIX_DEFAULTS):
            raise ValueError(f"unknown caption_mode: {self.caption_mode}")
        if self.mixed_weights is None:
            self.mixed_weights = MIX_DEFAULTS.get(self.caption_mode, MIX_DEFAULTS["mixed"]).copy()
        for k, weight in self.mixed_weights.items():
            if k not in CAPTION_VARIANTS:
                raise ValueError(f"unknown mixed_weights key: {k}")
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
                raise ValueError(f"mixed_weights.{k} must be finite and nonnegative")
        self.caption_mode = compatible_mix(self.caption_mode, self.mixed_weights)
        if self.caption_mode in MIX_DEFAULTS:
            allowed = MIX_DEFAULTS[self.caption_mode]
            for key, weight in self.mixed_weights.items():
                if weight and key not in allowed:
                    raise ValueError(f"{self.caption_mode} does not use mixed_weights.{key}")
            self.mixed_weights = {k: v for k, v in self.mixed_weights.items() if k in allowed}
        # GUI editors also serialize unused fields. Keep existing caption-cache identities.
        self.mixed_weights = {k: v for k, v in self.mixed_weights.items() if k not in MOSS_FALLBACK or v}
        if self.uses_moss:
            self.mixed_weights = {k: self.mixed_weights[k] for k in CAPTION_VARIANTS if k in self.mixed_weights}
        if not 0.0 <= self.tag_dropout_percent <= 1.0:
            raise ValueError("tag_dropout_percent must be in [0,1]")
        if not 0.0 <= self.caption_dropout_percent <= 1.0:
            raise ValueError("caption_dropout_percent must be in [0,1]")

    @classmethod
    def from_dict(cls, d: dict) -> CaptionConfig:
        d = dict(d)
        path = d.pop("protected_tags_file", None)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if path:
            cfg.protected_tags = load_protected_tags(path)
        return cfg

    def uses(self, variant):
        return self.caption_mode == variant or (self.caption_mode in MIX_DEFAULTS and self.mixed_weights.get(variant, 0) > 0)

    @property
    def uses_moss(self):
        return any(self.uses(k) for k in MOSS_FALLBACK)

    @property
    def layout_variants(self):
        if self.uses("layout_long"):
            return ("layout", "layout_long")  # Keep the medium layout available as a fallback.
        return ("layout",) if self.uses("layout") else ()

    def without_moss(self):
        if self.caption_mode in JSON_MIXES:
            return replace(self, caption_mode="tags", mixed_weights={})
        if not self.uses_moss:
            return self
        weights = dict.fromkeys(LEGACY_VARIANTS, 0)
        for k, v in self.mixed_weights.items():
            weights[MOSS_FALLBACK.get(k, k)] += v
        mode = "mixed" if self.caption_mode == "mixed_custom" else MOSS_FALLBACK.get(self.caption_mode, self.caption_mode)
        return replace(self, caption_mode=mode,
                       mixed_weights={k: v for k, v in weights.items() if v})


def load_protected_tags(path: str | Path) -> set[str]:
    """One tag per line; blank lines and `#` comments ignored."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"protected tags file not found: {p}")
    tags = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tags.add(line.lower())
    return tags


def split_tags(caption: str, delimiter: str = ", ") -> list[str]:
    sep = delimiter.strip() or ","
    return [t.strip() for t in caption.split(sep) if t.strip()]


def process_tags(tags: list[str], cfg: CaptionConfig, rng: random.Random) -> list[str]:
    """Apply dropout then shuffling, respecting protected and pinned-leading tags."""
    if not tags:
        return tags

    keep_n = min(cfg.shuffle_keep_first_n, len(tags))
    head, tail = tags[:keep_n], tags[keep_n:]

    if cfg.tag_dropout_percent > 0 and tail:
        # Leading pinned tags and protected tags are exempt from dropout.
        keepable = [t for t in tail if t.lower() in cfg.protected_tags]
        droppable = [t for t in tail if t.lower() not in cfg.protected_tags]

        n_drop = int(len(droppable) * cfg.tag_dropout_percent + 0.5)
        # Enforce the floor across the whole caption, not just the droppable subset.
        max_droppable = max(0, len(head) + len(keepable) + len(droppable) - cfg.min_tags_kept)
        n_drop = min(n_drop, max_droppable)

        if n_drop > 0:
            survivors = set(rng.sample(range(len(droppable)), len(droppable) - n_drop))
            droppable = [t for i, t in enumerate(droppable) if i in survivors]

        # Rebuild preserving original relative order.
        kept = set(keepable) | set(droppable)
        tail = [t for t in tail if t in kept]

    if cfg.shuffle_tags:
        rng.shuffle(tail)

    return head + tail


def process_nl(nl: str, cfg: CaptionConfig, rng: random.Random) -> str:
    """Optionally shuffle sentences; the first often carries framing/subject and can be pinned."""
    if not nl or not cfg.nl_shuffle_sentences:
        return nl

    parts = [s.strip() for s in nl.split(". ") if s.strip()]
    if len(parts) < 2:
        return nl

    if cfg.nl_keep_first_sentence:
        head, rest = parts[:1], parts[1:]
        rng.shuffle(rest)
        parts = head + rest
    else:
        rng.shuffle(parts)

    out = ". ".join(parts)
    return out if out.endswith(".") else out + "."


def select_variant(cfg: CaptionConfig, has_nl: bool, rng: random.Random) -> str:
    """Pick a caption form. Falls back to tags-only when the sample has no NL caption."""
    if cfg.caption_mode not in MIX_DEFAULTS:
        variant = cfg.caption_mode
    else:
        weights = {k: v for k, v in cfg.mixed_weights.items() if v > 0}
        if not weights:
            return "tags"
        keys = list(weights)
        variant = rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]

    if not has_nl and variant in ("nl", "tags_nl", "nl_tags"):
        return "tags"
    return variant


def shuffle_moss_prose(text: str, cfg: CaptionConfig, rng: random.Random) -> str:
    if not cfg.nl_shuffle_sentences:
        return text
    # Keep punctuation on its sentence, including closing quotes. Decimal points
    # have no following whitespace and stay within the sentence.
    sentences = []
    start = 0
    for match in _QUOTES_OR_BOUNDARIES.finditer(text):
        if match[0].isspace() and match.end() < len(text):
            sentences.append(text[start:match.start()])
            start = match.end()
    sentences.append(text[start:])
    first = int(cfg.nl_keep_first_sentence)
    if len(sentences) - first < 2:
        return text
    tail = sentences[first:]
    rng.shuffle(tail)
    return " ".join(sentences[:first] + tail)


def build_caption(
    tags_text: str,
    nl_text: str | None,
    cfg: CaptionConfig,
    rng: random.Random | None = None,
    moss: dict[str, str] | None = None,
) -> str:
    """Produce the final caption string for one sample.

    Returns "" when caption dropout fires — the unconditional sample the model needs for CFG.
    """
    rng = rng or random

    if cfg.caption_dropout_percent > 0 and rng.random() < cfg.caption_dropout_percent:
        return ""

    if moss is None:
        cfg = cfg.without_moss()
    variant = select_variant(cfg, bool(nl_text), rng)

    tags = process_tags(split_tags(tags_text, cfg.tag_delimiter), cfg, rng)
    tags_str = cfg.tag_delimiter.join(tags)

    if variant == "tags":
        return tags_str
    if variant in MOSS_FALLBACK:
        if variant in ("layout", "layout_long"):
            if variant in moss:
                return moss[variant]
            if "layout" in moss:
                return moss["layout"]
            return shuffle_moss_prose(moss["medium"], cfg, rng)
        if variant in ("tags_moss_long", "moss_long_tags"):
            prose = shuffle_moss_prose(moss["long"], cfg, rng)
            if not tags_str:
                return prose
            return f"{tags_str}. {prose}" if variant == "tags_moss_long" else f"{prose} {tags_str}"
        return shuffle_moss_prose(moss[variant.removeprefix("moss_")], cfg, rng)
    nl_str = process_nl(nl_text or "", cfg, rng)
    if variant == "nl":
        return nl_str
    if variant == "tags_nl":
        return f"{tags_str}. {nl_str}" if tags_str else nl_str
    if variant == "nl_tags":
        return f"{nl_str} {tags_str}" if tags_str else nl_str
    raise AssertionError(f"unreachable variant {variant}")


def read_caption_files(image_path: str | Path, *, read_nl=True) -> tuple[str, str | None]:
    """`<stem>.txt` -> tags, `<stem>_nl.txt` -> NL caption (None if absent)."""
    p = Path(image_path)
    tags_path = p.with_suffix(".txt")
    nl_path = p.with_name(p.stem + "_nl.txt")

    tags = tags_path.read_text(encoding="utf-8").strip() if tags_path.exists() else ""
    nl = nl_path.read_text(encoding="utf-8").strip() if read_nl and nl_path.exists() else None
    return tags, nl
