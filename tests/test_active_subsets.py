"""Disabled dataset folders must not be visited by preparation or cache placement."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import pytest
from PIL import Image

from trainer.data.caption_variations import caption_cache_path
from trainer.data.dataset import DatasetConfig, MageFlowDataset, SubsetConfig
from trainer.gui import bridge
from trainer.tools.cache_latents import config_cache_commands
from trainer.training.config import load_config
from trainer.ui_core.preferences import Preferences
from trainer.ui_core.session import Session


@pytest.fixture
def recipe(tmp_path):
    active = tmp_path / 'active'
    active.mkdir()
    Image.new('RGB', (256, 256)).save(active / 'one.png')
    flat = bridge.defaults() | {
        'train.model_family': 'moss_image', 'dataset.path': None,
        'dataset.source': 'encode',
        'dataset.subsets': [
            {'path': str(tmp_path / 'missing-target'), 'target_images': 0},
            {'path': str(tmp_path / 'missing-repeat'), 'num_repeats': 0},
            # An explicit positive target takes precedence even over zero repeats.
            {'path': str(active), 'num_repeats': 0, 'target_images': 2},
        ],
    }
    path = tmp_path / 'recipe.toml'
    bridge.write_toml(path, flat)
    return flat, path, active


def cache_args(path):
    return SimpleNamespace(config=str(path), resolution=None, batch_size=1,
                           devices='', overwrite=False, dry_run=True,
                           allow_missing_captions=True)


def test_filter_preserves_config_and_weighted_identity(recipe):
    flat, path, active = recipe
    cfg = load_config(path)
    before = asdict(cfg.dataset)
    assert len(cfg.dataset.effective_subsets()) == 3
    assert [s.path for s in cfg.dataset.active_subsets()] == [str(active)]
    assert asdict(cfg.dataset) == before
    ds = MageFlowDataset(cfg.dataset, require_cache=False)
    assert ds.weighted and len(ds.epoch_indices(0)) == 2
    assert bridge.dataset_paths(flat) == [str(active)]
    # A disabled row must still select the existing weighted layout for other rows.
    legacy = DatasetConfig(source='encode', subsets=[
        SubsetConfig('missing', num_repeats=0), SubsetConfig(str(active), num_repeats=2)])
    weighted = MageFlowDataset(legacy, require_cache=False)
    assert weighted.weighted and len(weighted.entries) == 1
    assert len(weighted.epoch_indices(0)) == 2


def test_cli_dry_run_skips_missing_disabled_folders(recipe):
    flat, path, active = recipe
    commands = config_cache_commands(cache_args(path))
    assert len(commands) == 1 and commands[0][5] == str(active)
    result = subprocess.run(commands[0], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (path.parent / 'missing-target').exists()
    assert not (path.parent / 'missing-repeat').exists()


def test_both_gui_preparation_paths_skip_disabled_folders(recipe, tmp_path, monkeypatch):
    from trainer.gui.app import TrainingGUI
    flat, path, active = recipe
    session = Session(root=tmp_path, preferences=Preferences(tmp_path / 'prefs.json'),
                      config_dir=tmp_path / 'configs', gpus=[dict(index=0, name='test')])
    try:
        session.flat = flat
        assert session._paths(flat) == TrainingGUI._dataset_paths(None, flat) == [str(active)]
        assert [job.launch.argv[5] for job in session.cache_jobs(flat)] == [str(active)]
        # Exercise Start, audit and queue preparation without launching a GPU worker.
        captured = []
        monkeypatch.setattr(session, '_launch_thread', lambda operation: operation())
        monkeypatch.setattr(session, '_sequence', lambda jobs, *args: captured.extend(jobs))
        session.start('audit')
        assert [job.launch.argv[5] for job in captured] == [str(active)]
        captured.clear()
        session.start('train')
        caches = [j for j in captured if 'trainer.tools.cache_latents' in j.launch.argv]
        assert [j.launch.argv[5] for j in caches] == [str(active)]
        session.add_queue([path])
        _, jobs = session._prepare_entry(session.entries[0])
        caches = [j for j in jobs if 'trainer.tools.cache_latents' in j.launch.argv]
        assert [j.launch.argv[5] for j in caches] == [str(active)]
    finally:
        session.close()


@pytest.mark.parametrize('setting', [{'num_repeats': 0}, {'num_repeats': 3, 'target_images': 0}])
def test_all_disabled_fails_validation_and_cli(tmp_path, setting):
    flat = bridge.defaults() | {'dataset.path': str(tmp_path / 'missing')}
    flat.update({f'dataset.{k}': v for k, v in setting.items()})
    path = tmp_path / 'disabled.toml'
    bridge.write_toml(path, flat)
    assert bridge.dataset_paths(flat) == []
    valid, error = bridge.validate(flat)
    assert not valid and 'empty dataset' in error
    with pytest.raises(ValueError, match='empty dataset'):
        config_cache_commands(cache_args(path))


def test_caption_cache_uses_active_folder_and_respects_override(recipe):
    _, path, active = recipe
    cfg = load_config(path)
    assert Path(caption_cache_path(cfg)) == active / 'cached_text_embeddings/caption_variations.sqlite'
    cfg.train.caption_cache_path = str(path.parent / 'custom' / 'captions.sqlite')
    assert caption_cache_path(cfg) == cfg.train.caption_cache_path
    assert not (path.parent / 'missing-target').exists()


def test_eye_detection_only_visits_active_sources(recipe, tmp_path, monkeypatch):
    from trainer.tools import detect_eyes
    _, config, active = recipe
    model = tmp_path / 'fake-model.pt'
    model.write_bytes(b'fake model')
    calls = []
    result = SimpleNamespace(masks=None, boxes=[])
    fake = SimpleNamespace(task='segment', names={},
                           predict=lambda *a, **kw: calls.append(1) or [result])
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=lambda p: fake, __version__='test'))
    rglob = Path.rglob
    def visit(folder, pattern):
        assert folder == active, f'Disabled source was visited: {folder}'
        return rglob(folder, pattern)
    monkeypatch.setattr(Path, 'rglob', visit)
    output = tmp_path / 'eyes'
    monkeypatch.setattr(sys, 'argv', ['detect_eyes', str(config), '--model', str(model), '--output', str(output)])
    detect_eyes.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert [row['source'] for row in manifest['images']] == [str(active / 'one.png')]
    assert len(calls) == 2  # Source plus the single training tier, never repeated.
    assert not (tmp_path / 'missing-target').exists()
