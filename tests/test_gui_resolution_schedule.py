import json
import os
import sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtWidgets

from trainer.gui import bridge
from trainer.gui.fields import ResolutionStagesEditor
from trainer.gui.metrics import LiveMetricsWidget
from test_resolution_schedule import config_text
from test_gui_dataset_experiments import gui


def test_bridge_staged_round_trip_omits_epochs(tmp_path):
    import tomllib
    from trainer.training.config import load_config
    flat = {**bridge.defaults(), **bridge.flatten(tomllib.loads(config_text()))}
    if sys.platform == 'win32':
        flat['train.attention_backend'] = 'flash_attn_2'
    flat['train.epochs'] = 100
    flat['train.save_every_epochs'] = 2
    text = bridge.dump_toml(flat)
    assert 'epochs' not in text
    p = tmp_path/'staged.toml'
    p.write_text(text)
    cfg = load_config(p)
    assert cfg.resolution_schedule.budgets(cfg.train.max_steps) == [112,112,336]
    assert [s.lr_mul for s in cfg.resolution_schedule.stages] == [.8, .9, 1.0]
    assert bridge.validate(flat)[0]
    flat['resolution_schedule.enabled'] = False
    assert 'epochs = 100' in bridge.dump_toml(flat)


def test_stage_editor_budget_and_batch_preview():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    editor = ResolutionStagesEditor()
    stages = [dict(resolution=r,share=s,lr_mul=lr) for r,s,lr in ((512,.2,.8),(768,.2,.9),(1024,.6,1.))]
    editor.set(stages)
    assert editor.get() == stages
    editor.preview(560,{512:24,768:12,1024:7},2,1)
    assert [editor.table.item(r,4).text() for r in range(3)] == ['112','112','336']
    assert editor.table.item(0,5).text() == '48'
    editor.widget.deleteLater()
    app.processEvents()


def test_stage_metrics_reach_graph_and_header():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = LiveMetricsWidget()
    widget.parse_and_update('stage2 step 113/560 loss 0.3 lr 1e-4 2s/it peak_alloc 9.0GiB')
    widget.parse_and_update('BLUFLOW_STAGE ' + json.dumps(dict(stage=2,resolution=768,completed=1,total=112,images=24,effective_batch=24)))
    widget._flush()
    assert widget.latest['stage'] == 2
    assert 'epoch' not in widget.latest
    assert widget.latest['step'] == 113
    assert '768px' in widget.header.text()
    widget.deleteLater()
    app.processEvents()


def test_gui_enabling_stages_hides_epochs_and_seeds_valid_plan(gui):
    gui._apply(bridge.defaults() | {
        'dataset.path': '/images', 'dataset.source': 'latents', 'train.model_family': 'moss_image',
        'adapter.kind': 'lora', 'train.batch_size': 2,
        'dataset.resolutions': [256, 384, 512, 640, 768, 1024],
    })
    gui.editors['resolution_schedule.enabled'].set(True)
    gui._on_edit()
    flat = gui.collect()
    assert flat['train.batch_size'] == {r: 2 for r in [256, 384, 512, 640, 768, 1024]}
    assert flat['train.pack_resolutions']
    assert flat['train.max_steps'] == 560
    assert abs(sum(s['share'] for s in flat['resolution_schedule.stages']) - 1) < 1e-9
    assert {s['resolution']: s['lr_mul'] for s in flat['resolution_schedule.stages']} == {
        256: 1., 384: 1., 512: .8, 640: 1., 768: .9, 1024: 1.}
    assert all(w.isHidden() for w in gui.rows['train.epochs'])
    assert not gui.editors['dataset.tier_collapse'].widget.isEnabled()
    ok, error = bridge.validate(flat)
    assert ok, error
    gui.editors['resolution_schedule.enabled'].set(False)
    gui._on_edit()
    assert not any(w.isHidden() for w in gui.rows['train.epochs'])


def test_epoch_duration_and_weight_switch_round_trip(gui, tmp_path):
    from trainer.training.config import load_config
    gui._apply(bridge.defaults() | {
        'dataset.path': '/images', 'dataset.source': 'latents', 'train.model_family': 'moss_image',
        'adapter.kind': 'lora', 'train.batch_size': 2,
    })
    gui.editors['resolution_schedule.enabled'].set(True)
    gui._on_edit()
    gui.editors['resolution_schedule.unit'].set('epochs')
    gui.editors['train.epochs'].set(50)
    gui.editors['train.save_every_epochs'].set(2)
    gui.editors['dataset.num_repeats'].set(1.56)
    gui._on_edit()
    assert not any(w.isHidden() for w in gui.rows['train.epochs'])
    assert all(w.isHidden() for w in gui.rows['train.max_steps'])
    stages = gui.editors['resolution_schedule.stages']
    assert [stages.table.item(r, 4).text() for r in range(3)] == ['10', '10', '30']
    target = gui.editors['dataset.target_images']
    target.toggle.setChecked(True)
    target.count.setValue(156)
    gui._on_edit()
    assert gui.collect()['dataset.target_images'] == 156
    assert all(w.isHidden() for w in gui.rows['dataset.num_repeats'])
    path = tmp_path / 'epoch-target.toml'
    bridge.write_toml(path, gui.collect())
    assert 'num_repeats' not in path.read_text()
    assert 'max_steps' not in path.read_text()
    cfg = load_config(path)
    assert cfg.dataset.target_images == 156
    assert cfg.train.epochs == 50 and cfg.train.save_every_epochs == 2
    target.toggle.setChecked(False)
    gui._on_edit()
    assert gui.collect()['dataset.num_repeats'] == 1.56


def test_subset_weight_switch_round_trip():
    from trainer.gui.fields import SubsetEditor
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    editor = SubsetEditor()
    rows = [dict(path='/a', num_repeats=1.56), dict(path='/b', target_images=25, texture=False)]
    editor.set(rows)
    assert editor.get() == rows
    target = editor.table.cellWidget(0, 3).editor
    target.toggle.setChecked(True)
    target.count.setValue(75)
    assert editor.get()[0] == dict(path='/a', target_images=75)
    assert not editor.table.cellWidget(0, 1).isEnabled()
    target.toggle.setChecked(False)
    assert editor.get()[0] == rows[0]
    editor.widget.deleteLater()
    app.processEvents()
