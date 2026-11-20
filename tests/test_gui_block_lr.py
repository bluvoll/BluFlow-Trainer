import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

from trainer.gui import bridge
from trainer.training.config import load_config
from test_gui_dataset_experiments import gui


@pytest.mark.parametrize('kind', ['lora', 'none', 'lokr', 'lycoris_lokr'])
def test_experimental_block_table_roundtrip(gui, tmp_path, kind):
    flat = bridge.defaults() | {'dataset.path':'/images', 'train.model_family':'moss_image', 'adapter.kind':kind}
    if kind == 'lycoris_lokr':
        flat.update({'adapter.kind': 'lycoris_lora', 'adapter.lycoris_algo': 'lokr'})
    gui._apply(flat)
    editor = gui.editors['block_lr.ranges']
    assert gui.editors['block_lr.enabled'].widget.isEnabled()
    assert not editor.widget.isEnabled()
    assert editor.get() == flat['block_lr.ranges']
    gui.editors['block_lr.enabled'].set(True)
    gui._on_edit()
    assert gui.tab_bar.isTabVisible(gui.experimental_tab)
    assert editor.widget.isEnabled()
    editor.table.cellWidget(0, 2).setValue(.75)
    editor.table.cellWidget(2, 2).setValue(1.5)
    config = gui.collect()
    ok, error = bridge.validate(config)
    assert ok, error
    path = tmp_path/'blocks.toml'
    bridge.write_toml(path, config)
    assert load_config(path).block_lr.expanded() == [.75]*25+[1.]*8+[1.5]*9
    gui._apply(bridge.defaults() | bridge.flatten(bridge.read_toml(path)))
    assert gui.collect()['block_lr.ranges'] == config['block_lr.ranges']
    assert gui.tab_bar.isTabVisible(gui.experimental_tab)
    gui.editors['block_lr.enabled'].set(False)
    gui._on_edit()
    bridge.write_toml(path, gui.collect())
    cfg = load_config(path)
    assert cfg.block_lr.expanded() == [1.]*42
    assert cfg.block_lr.ranges[0]['lr_mul'] == .75


def test_overlaps_do_not_silently_change_and_unsupported_can_disable(gui):
    gui._apply(bridge.defaults() | {'dataset.path':'/images', 'train.model_family':'moss_image', 'adapter.kind':'lora', 'block_lr.enabled':True})
    editor = gui.editors['block_lr.ranges']
    editor.table.cellWidget(1,0).setValue(24)
    ok, error = bridge.validate(gui.collect())
    assert not ok and 'overlap' in error
    editor.table.cellWidget(1,0).setValue(25)
    editor.table.selectRow(1)
    editor._remove()
    editor._add()
    assert editor.get()[-1] == dict(start=25,end=25,lr_mul=1.)
    gui.editors['adapter.kind'].set('lycoris_lora')
    gui.editors['adapter.lycoris_algo'].set('dora')
    gui._on_edit()
    assert gui.editors['block_lr.enabled'].widget.isEnabled()
    ok, error = bridge.validate(gui.collect())
    assert not ok and 'block LR' in error


def test_depth_settings_roundtrip_and_survive_disable(gui, tmp_path):
    target = .400000001
    rows = [dict(start=0, end=3, lr_mul=target),
            dict(start=4, end=5, lr_mul=.8, scale_by_depth=True)]
    gui._apply(bridge.defaults() | {'dataset.path': '/images', 'train.model_family': 'moss_image',
                                    'block_lr.enabled': True, 'block_lr.ranges': rows})
    minimum = gui.editors['block_lr.min_lr_mul']
    ranges = gui.editors['block_lr.ranges']
    assert minimum.get() == .1
    minimum.set(.15)
    ranges.table.cellWidget(0, 3).setChecked(True)
    cfg = gui.collect()
    path = tmp_path / 'depth.toml'
    bridge.write_toml(path, cfg)
    saved = load_config(path).block_lr
    assert saved.min_lr_mul == .15
    assert saved.expanded()[:6] == pytest.approx([.15, .23333333367, .31666666733, target, target, .8])
    assert saved.ranges[0]['lr_mul'] == target
    gui._apply(bridge.defaults() | bridge.flatten(bridge.read_toml(path)))
    assert ranges.get() == cfg['block_lr.ranges']
    gui.editors['block_lr.enabled'].set(False)
    gui._on_edit()
    assert not minimum.widget.isEnabled()
    bridge.write_toml(path, gui.collect())
    disabled = load_config(path).block_lr
    assert disabled.expanded() == [1.] * 42
    assert disabled.ranges == saved.ranges
    assert disabled.min_lr_mul == saved.min_lr_mul
