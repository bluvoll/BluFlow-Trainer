"""Cabal-style source weighting, before tiers and independent of loader/rank state."""
from collections import Counter
from dataclasses import replace

import pytest
from PIL import Image

from trainer.data.dataset import DatasetConfig, MageFlowDataset, SubsetConfig, BucketBatchSampler
from trainer.data.packed import NativeResolutionBatchSampler


@pytest.fixture
def images(tmp_path):
    for i in range(100):
        path = tmp_path / f'{i:03}.png'
        Image.new('RGB', (256, 256)).save(path)
        path.with_suffix('.txt').write_text('red, green, blue, orange, purple')
    return DatasetConfig(path=str(tmp_path), resolutions=[64, 128], min_bucket_reso=64,
                         bucket_reso_steps=64, bucket_no_upscale=False)


@pytest.mark.parametrize('setting,count', [
    ({'num_repeats': 1.56}, 156), ({'num_repeats': .5}, 50),
    ({'target_images': 50}, 50), ({'target_images': 250}, 250),
    ({'num_repeats': 2.0}, 200),
])
def test_counts_and_resolution_selection(images, setting, count):
    ds = MageFlowDataset(replace(images, **setting), require_cache=False, caption_seed=42)
    for epoch in (0, 1, 12):
        selections = []
        for tier in images.tiers:
            indices = ds.epoch_indices(epoch, tier)
            assert len(indices) == count
            selections.append(Counter(ds.entries[i].source_key for i in indices))
        assert selections[0] == selections[1]


def test_rotates_and_replays_without_mutating_candidates(images):
    ds = MageFlowDataset(replace(images, target_images=40), require_cache=False, caption_seed=42)
    original = list(ds.entries)
    first = ds.epoch_indices(0)
    assert ds.epoch_indices(1) != first
    assert ds.epoch_indices(0) == first
    assert ds.entries == original
    other = MageFlowDataset(replace(images, target_images=40), require_cache=False, caption_seed=42)
    assert other.epoch_indices(1) == ds.epoch_indices(1)


@pytest.mark.parametrize('sampler_cls', [BucketBatchSampler, NativeResolutionBatchSampler])
def test_repeated_appearances_have_distinct_stable_visits(images, sampler_cls):
    ds = MageFlowDataset(replace(images, num_repeats=1.56), require_cache=False, caption_seed=42)
    sampler = sampler_cls(ds, 7, seed=42)
    samples = [s for b in sampler for s in b]
    assert len(samples) == 312
    assert len({(s[0], s[4]) for s in samples}) == 312
    assert list(sampler) == list(sampler)
    sampler.set_epoch(1)
    assert [s for b in sampler for s in b] != samples


def test_zero_subset_is_disabled_and_empty_dataset_rejected(images):
    cfg = replace(images, path=None, subsets=[SubsetConfig('does-not-exist', target_images=0),
                                              SubsetConfig(images.path, num_repeats=.5)])
    ds = MageFlowDataset(cfg, require_cache=False)
    assert len(ds.epoch_indices(0)) == 100
    with pytest.raises(ValueError, match='empty dataset'):
        MageFlowDataset(replace(images, target_images=0), require_cache=False)


@pytest.mark.parametrize('kwargs', [
    {'num_repeats': -1}, {'num_repeats': float('nan')}, {'num_repeats': float('inf')},
    {'num_repeats': True}, {'target_images': 1.56}, {'target_images': -1}, {'target_images': True},
])
def test_invalid_weights(kwargs):
    with pytest.raises(ValueError):
        DatasetConfig(path='unused', **kwargs)


def test_cache_only_source_selection_matches_images(tmp_path):
    from trainer.parity.test_multires import make_fixture
    make_fixture(tmp_path, [(256, 256), (1200, 1200), (1400, 800), (800, 1400)], [512, 768])
    cfg = DatasetConfig(path=str(tmp_path), source='images', resolutions=[512, 768],
                        tier_collapse='repeat', target_images=7)
    image_ds = MageFlowDataset(cfg, caption_seed=42)
    for path in tmp_path.glob('*.png'):
        path.unlink()
    cached_ds = MageFlowDataset(replace(cfg, source='latents'), caption_seed=42)
    def selection(ds, epoch):
        return [(ds.entries[i].source_key, ds.entries[i].resolution) for i in ds.epoch_indices(epoch)]
    for epoch in range(3):
        assert selection(image_ds, epoch) == selection(cached_ds, epoch)
        assert len(selection(cached_ds, epoch)) == 14


def test_caption_cache_visits_remain_stable_for_fractional_repeats(images, tmp_path):
    from trainer.data.caption import CaptionConfig
    from trainer.data.caption_variations import CaptionVariationCache
    cfg = replace(images, source='encode', num_repeats=1.56, caption=CaptionConfig(shuffle_tags=True))
    ds = MageFlowDataset(cfg, require_cache=False)
    cache = CaptionVariationCache(tmp_path / 'captions.sqlite', 'test')
    cache.prepare(ds.entries, cfg.caption, 42, 100)
    ds.caption_variation_cache = cache
    before = [ds[(0, 0, None, None, visit)]['caption'] for visit in range(20)]
    assert len(set(before)) > 1
    ds.epoch_indices(7)
    assert before == [ds[(0, 0, None, None, visit)]['caption'] for visit in range(20)]
    cache.close()
