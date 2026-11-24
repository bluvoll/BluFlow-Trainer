"""Build MOSS text conditioning from tagpipe sidecars and each entry's actual crop."""

import json
import warnings
from collections import Counter

from .cache import parse_original_size, read_cache_metadata


def layout_options(cfg):
    if not cfg.dataset.caption.layout_variants:
        return {}
    from ..modeling.moss_loader import text_sources
    _, tokenizer = text_sources(cfg.train.model_path, tokenizer_path=cfg.train.tokenizer_path)
    return {"layout_tokenizer_path": tokenizer, "max_text_tokens": cfg.train.max_text_tokens}


def compile_layout(layout, summary, resized, crop):
    """Map normalized source edges into a clipped, center-origin pixel box."""
    left, top, right, bottom = crop
    width, height = right - left, bottom - top
    rw, rh = resized
    if not isinstance(layout, dict) or not isinstance(layout.get("canvas"), str) or not layout["canvas"].strip():
        raise ValueError("layout needs a canvas description")
    items = []
    for item in layout["items"]:
        box = item["box_2d"]
        if len(box) != 4 or any(type(v) is not int or not 0 <= v <= 1000 for v in box):
            raise ValueError(f"invalid normalized box {box}")
        y0, x0, y1, x1 = box
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"empty or reversed box {box}")
        # Work in thousandths of a resized pixel to round fractional edges only once.
        x0, x1 = x0 * rw - 1000 * left, x1 * rw - 1000 * left
        y0, y1 = y0 * rh - 1000 * top, y1 * rh - 1000 * top
        area = (x1 - x0) * (y1 - y0)
        x0, x1 = max(0, x0), min(width * 1000, x1)
        y0, y1 = max(0, y0), min(height * 1000, y1)
        visible = max(0, x1 - x0) * max(0, y1 - y0)
        label, caption = item["label"], item["caption"]
        if not all(isinstance(s, str) and s.strip() for s in (label, caption)):
            raise ValueError("layout item needs a label and caption")
        if visible * 2 < area or (label == "text" and visible < area):
            continue
        bbox = [y0 // 1000 - height // 2, x0 // 1000 - width // 2,
                (y1 + 999) // 1000 - height // 2, (x1 + 999) // 1000 - width // 2]
        out = {"label": label, "caption": caption, "bbox": bbox}
        if label == "text":
            if not isinstance(item.get("text"), str) or not item["text"].strip():
                raise ValueError("layout text item needs a transcription")
            out["text"] = item["text"]
        items.append(out)
    if not items:
        return None
    return json.dumps({"bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
                       "summary": {"canvas": layout["canvas"], "caption": summary}, "items": items},
                      ensure_ascii=False, separators=(",", ":"))


def entry_geometry(entry, manager, use_cached):
    if use_cached and entry.latent_path.is_file():
        meta = read_cache_metadata(entry.latent_path)
        size = parse_original_size(meta)
        if not size or not meta.get("crop_ltrb"):
            return None
        crop = tuple(map(int, meta["crop_ltrb"].split(",")))
    else:
        size, crop = entry.source_size, None
    if not all(size):
        return None
    bucket, (rw, rh), _ = manager.select_bucket(*size)
    w, h = bucket
    if bucket != entry.bucket:
        raise ValueError(f"selected bucket {bucket} differs from entry {entry.bucket}")
    if crop is None:
        left, top = (rw - w) // 2, (rh - h) // 2
        crop = (left, top, left + w, top + h)
    l, t, r, b = crop
    if (r - l, b - t) != bucket or not (0 <= l < r <= rw and 0 <= t < b <= rh):
        raise ValueError(f"invalid latent crop {crop} for resized image {(rw, rh)}")
    return (rw, rh), crop


def attach_moss_captions(entries, managers, *, layout_variants, use_cached, tokenizer_path, max_tokens):
    from ..modeling.moss_loader import ASSETS, format_prompt

    sources = {}
    tokenizer = None
    fallbacks = Counter()
    for entry in entries:
        path = entry.path.with_name(f"{entry.source_key}_moss.json")
        try:
            if path not in sources:
                document = None
                if path.is_file():
                    document = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(document, dict):
                        raise ValueError("expected a MOSS sidecar object")
                sources[path] = document
            document = sources[path]
            if document is None:
                continue  # Ordinary datasets, including color regularization, use their .txt captions.
            prose = {k: document["caption"][k] for k in ("short", "medium", "long")}
            if not all(isinstance(v, str) and v.strip() for v in prose.values()):
                raise ValueError("short, medium and long captions must be nonempty strings")
            entry.moss = prose
            if not layout_variants:
                continue
            if not document.get("layout"):
                fallbacks["missing layout"] += 1
                continue
            geometry = entry_geometry(entry, managers[entry.resolution], use_cached)
            if geometry is None:
                fallbacks["unknown crop"] += 1
                continue
            for variant in layout_variants:
                summary = prose["long" if variant == "layout_long" else "medium"]
                prompt = compile_layout(document["layout"], summary, *geometry)
                if not prompt:
                    fallbacks["empty after crop"] += 1
                    break
                if tokenizer is None:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or ASSETS, local_files_only=True)
                tokens = tokenizer(format_prompt(tokenizer, prompt), truncation=False, verbose=False).input_ids
                if len(tokens) <= max_tokens:
                    prose[variant] = prompt
                else:
                    fallbacks[f"{variant} over token limit"] += 1
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path} ({entry.bucket}): {error}") from error
    if fallbacks:
        warnings.warn(f"MOSS layout fallback counts by image/bucket: {dict(fallbacks)}",
                      stacklevel=2)
