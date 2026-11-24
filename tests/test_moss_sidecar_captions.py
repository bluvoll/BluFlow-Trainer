import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainer.data.caption import CaptionConfig, build_caption
from trainer.data.caption_variations import CaptionVariationCache
from trainer.data.dataset import DatasetConfig, MageFlowDataset
from trainer.data.moss_captions import compile_layout


def test_caption_mode_separation_and_old_mix_import():
    from trainer.gui import bridge
    from trainer.data.text_cache import validate_static_captions
    caption = "dataset.caption."
    for old_key, new_mode in [("moss_medium", "mixed_json"), ("layout", "mixed_json_layout")]:
        weights = {"tags": 35, old_key: 65}
        flat = bridge.flatten({"dataset": {"caption": {"caption_mode": "mixed", "mixed_weights": weights}}})
        assert flat[caption + "caption_mode"] == new_mode
        assert flat[caption + "mixed_weights"] == weights
        cfg = CaptionConfig(caption_mode="mixed", mixed_weights=weights)
        assert cfg.caption_mode == new_mode
        assert build_caption("tags only", "unused", cfg) == "tags only"
        with pytest.raises(ValueError, match=new_mode):
            validate_static_captions(SimpleNamespace(dataset=SimpleNamespace(caption=cfg), train=None))
    hybrid = CaptionConfig(caption_mode="mixed", mixed_weights={"nl": 2, "layout": 1})
    assert hybrid.caption_mode == "mixed_custom" and hybrid.uses("nl") and hybrid.uses("layout")
    assert not CaptionConfig(caption_mode="mixed").uses_moss
    with pytest.raises(ValueError, match="layout"):
        CaptionConfig(caption_mode="mixed_json", mixed_weights={"tags": 1, "layout": 1})


@pytest.mark.parametrize("variant", ["moss_short", "moss_medium", "moss_long", "tags_moss_long", "moss_long_tags", "layout"])
def test_json_prose_shuffles_whole_sentences_and_pins_first(variant):
    prose = 'Scene. A sign reads "Yes! Come in." Rain falls! All clear?'
    moss = dict.fromkeys(("short", "medium", "long"), prose)
    cfg = CaptionConfig(caption_mode=variant, nl_shuffle_sentences=True, nl_keep_first_sentence=True)
    texts = {build_caption("coat, scarf", "ignored", cfg, random.Random(i), moss) for i in range(16)}
    assert len(texts) > 1
    for text in texts:
        if variant == "tags_moss_long":
            assert text.startswith("coat, scarf. Scene.")
        else:
            assert text.startswith("Scene.")
        if variant == "moss_long_tags":
            assert text.endswith(" coat, scarf")
        assert 'A sign reads "Yes! Come in."' in text
        assert "Rain falls!" in text and "All clear?" in text
    cfg.nl_shuffle_sentences = False
    assert prose in build_caption("coat, scarf", None, cfg, random.Random(0), moss)


def test_layout_crops_and_clips_in_resized_pixels():
    item = dict(label="character", caption="left", box_2d=[0, 0, 1000, 500])
    text = dict(label="text", caption="sign", text="AB", box_2d=[200, 0, 400, 300])
    other = dict(label="object", caption="edge", box_2d=[200, 0, 400, 100])
    layout = dict(canvas="single", items=[item, text, other])
    cropped = json.loads(compile_layout(layout, "test", (160, 100), (20, 0, 140, 100)))
    assert cropped["bbox_format"] == "yxyx"
    assert cropped["bbox_coord_space"] == "abs_centralized"
    assert cropped["items"] == [{"label": "character", "caption": "left", "bbox": [-50, -60, 50, 0]}]
    uncropped = json.loads(compile_layout(layout, "test", (160, 100), (0, 0, 160, 100)))
    assert uncropped["items"][1]["text"] == "AB"
    with pytest.raises(ValueError, match="reversed"):
        compile_layout(dict(canvas="single", items=[dict(item, box_2d=[0, 0, 50, 0])]),
                       "test", (100, 100), (0, 0, 100, 100))


def test_prose_layout_match_visits_dropout_and_shared_captions(tmp_path):
    moss = dict(medium="medium, then more.", long="Start. Next. Last.", short="One.")
    entries = [
        SimpleNamespace(path=Path("style/a.jpg"), tags="c, a, b", nl=None, moss=dict(moss, layout=layout))
        for layout in ("layout-512", "layout-1280")
    ]
    # This stem belongs to a second, ordinary subset without MOSS sidecars.
    entries.append(SimpleNamespace(path=Path("color/b.jpg"), tags="red, solid", nl="Red.", moss=None))
    caches = []
    for variant in ("moss_medium", "layout"):
        cfg = CaptionConfig(caption_mode="mixed", mixed_weights={variant: 50, "tags_moss_long": 50},
                            shuffle_tags=True, nl_shuffle_sentences=True, caption_dropout_percent=0.1)
        cache = CaptionVariationCache(tmp_path / f"{variant}.sqlite", "test")
        cache.prepare(entries, cfg, 42, 12)
        cache.close()
        reader = CaptionVariationCache(tmp_path / f"{variant}.sqlite", "test")
        reader.prepare(entries, cfg, 42, 12, write=False)
        caches.append(reader)
    try:
        for i, entry in enumerate(entries):
            different = set()
            for epoch in range(48):
                a, b = [cache.caption(i, epoch, occurrence=2) for cache in caches]
                if a != b:
                    assert a == moss["medium"]
                    different.add(b)
            assert different == ({entry.moss["layout"]} if entry.moss else set())
        layout = CaptionConfig(caption_mode="layout", nl_shuffle_sentences=True)
        assert build_caption("a", None, layout, random.Random(0), moss) == moss["medium"]
        assert build_caption("a", None, layout) == "a"
        assert build_caption("a", "legacy prose", layout) == "legacy prose"
    finally:
        for cache in caches:
            cache.close()


def test_moss_shuffle_replans_existing_unshuffled_slots(tmp_path):
    from dataclasses import asdict
    from trainer.data.caption_variations import digest
    entry = SimpleNamespace(path=tmp_path / "a.jpg", tags="tag", nl=None,
                            moss={"long": "First. Second! Third?"})
    cfg = CaptionConfig(caption_mode="moss_long", nl_shuffle_sentences=True)
    cache = CaptionVariationCache(tmp_path / "variations.sqlite", "test")
    identity = (str(tmp_path), "a", "tag", None)
    sampling = digest((identity, 42, "moss-caption-draw-v1"))
    old_pool = digest((sampling, asdict(cfg), entry.moss))
    with cache.writer() as db:
        key = digest(entry.moss["long"])
        db.execute("INSERT INTO captions VALUES (?, ?)", (key, entry.moss["long"]))
        db.executemany("INSERT INTO slots VALUES (?, ?, ?)", [(old_pool, i, key) for i in range(12)])
    db.close()
    try:
        cache.prepare([entry], cfg, 42, 12)
        new_pool, _ = cache.indices[0]
        assert new_pool != old_pool
        expected = {build_caption(entry.tags, None, cfg, random.Random(digest((sampling, i))), entry.moss)
                    for i in range(12)}
        assert len(expected) > 1
        assert {cache.caption(0, epoch) for epoch in range(12)} == expected
    finally:
        cache.close()


def test_dataset_reads_sidecars_and_checks_template_length(tmp_path):
    from PIL import Image

    Image.new("RGB", (192, 128)).save(tmp_path / "image.png")
    (tmp_path / "image.txt").write_text("tag")
    doc = {"caption": {"short": "short", "medium": "medium", "long": "long"},
           "layout": {"canvas": "single", "items": [
               {"label": "text", "caption": "a sign", "text": "HELLO", "box_2d": [100, 200, 400, 800]}]}}
    (tmp_path / "image_moss.json").write_text(json.dumps(doc))
    cfg = DatasetConfig(path=str(tmp_path), source="images", resolution=128, caption=CaptionConfig(caption_mode="layout"))
    ds = MageFlowDataset(cfg, require_cache=False)
    assert json.loads(ds.entries[0].moss["layout"])["items"][0]["text"] == "HELLO"
    with pytest.warns(UserWarning, match="over token limit"):
        ds = MageFlowDataset(cfg, require_cache=False, max_text_tokens=16)
    assert "layout" not in ds.entries[0].moss
    assert build_caption("tag", None, cfg.caption, moss=ds.entries[0].moss) == "medium"

    doc["caption"]["long"] = "A long sentence. " * 200
    (tmp_path / "image_moss.json").write_text(json.dumps(doc))
    (tmp_path / "image_nl.txt").write_bytes(b"\xff")
    cfg.caption = CaptionConfig(caption_mode="mixed_json_layout", mixed_weights={"layout_long": 100},
                                nl_shuffle_sentences=True)
    with pytest.warns(UserWarning, match="layout_long over token limit"):
        ds = MageFlowDataset(cfg, require_cache=False)
    entry = ds.entries[0]
    assert entry.nl is None
    assert "layout_long" not in entry.moss
    assert build_caption("tag", None, cfg.caption, moss=entry.moss) == entry.moss["layout"]
    doc["caption"]["long"] = "A fuller scene description. Here is a sign."
    (tmp_path / "image_moss.json").write_text(json.dumps(doc))
    entry = MageFlowDataset(cfg, require_cache=False).entries[0]
    prompt = build_caption("tag", None, cfg.caption, moss=entry.moss)
    expected = json.loads(entry.moss["layout"])
    expected["summary"]["caption"] = doc["caption"]["long"]
    assert json.loads(prompt) == expected

    # Prose-only training must work even if the file carries unusable layout data.
    doc["layout"] = {"items": "broken"}
    (tmp_path / "image_moss.json").write_text(json.dumps(doc))
    cfg.caption = CaptionConfig(caption_mode="mixed_json")
    entry = MageFlowDataset(cfg, require_cache=False).entries[0]
    assert set(entry.moss) == {"short", "medium", "long"}
