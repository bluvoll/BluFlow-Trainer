import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
import toml
from PySide6 import QtWidgets
from trainer.gui import bridge
from trainer.gui.app import TrainingGUI
from trainer.gui.preferences import Preferences
from trainer.training.config import load_config
from trainer.training.quant import QUANT_PRESETS


@pytest.mark.parametrize('preset',['standard','balanced','fast'])
@pytest.mark.parametrize('kind,algo',[('lycoris_lora','locon'),('lora','locon'),('lokr','locon'),('lycoris_lora','lokr')])
def test_preset_controls_save_and_reload(monkeypatch,tmp_path,preset,kind,algo):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(TrainingGUI,'_load_presets',lambda self,**kw:None)
    gui=TrainingGUI(preferences=Preferences(tmp_path/"gui.json"), config_dir=str(tmp_path))
    try:
        gui._apply(bridge.defaults() | {'dataset.path':'/images','train.model_family':'moss_image',
            'adapter.kind':kind,'adapter.lycoris_algo':algo,'adapter.dtype':'bfloat16'})
        gui.editors['quant.preset'].set(preset);gui._on_edit()
        flat=gui.collect()
        for key,value in QUANT_PRESETS[preset].items():
            assert flat[f'quant.{key}']==value
            assert not gui.editors[f'quant.{key}'].widget.isEnabled()
        path=tmp_path/'preset.toml';path.write_text(toml.dumps(bridge.nest(bridge.prune_defaults(flat))))
        cfg=load_config(path);assert cfg.quant.preset==preset
        gui._apply(bridge.defaults() | bridge.flatten(bridge.read_toml(path)))
        assert gui.collect()['quant.preset']==preset
        gui.editors['quant.preset'].set('custom');gui._on_edit()
        assert gui.editors['quant.mode'].widget.isEnabled()
        assert gui.editors['quant.weights_dtype'].widget.isEnabled()
        assert gui.collect()['quant.weights_dtype']==cfg.quant.weights_dtype
    finally:
        gui.close();gui.deleteLater();app.processEvents()


def test_full_finetune_hadamard_controls(monkeypatch,tmp_path):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(TrainingGUI,'_load_presets',lambda self,**kw:None)
    gui=TrainingGUI(preferences=Preferences(tmp_path/'gui.json'))
    try:
        gui._apply(bridge.defaults()|{'dataset.path':'/images','train.model_family':'moss_image',
            'adapter.kind':'none','train.dtype':'bfloat16','quant.preset':'custom','quant.mode':'training',
            'quant.weights_dtype':'int8','quant.group_size':-1,'quant.use_quantized_matmul':True})
        assert gui.editors['quant.use_hadamard'].widget.isEnabled()
        gui.editors['quant.use_hadamard'].set(True);gui._on_edit()
        assert bridge.validate(gui.collect())[0]
        path=tmp_path/'full-hadamard.toml';bridge.write_toml(path,gui.collect())
        assert load_config(path).quant.use_hadamard
    finally:
        gui.close();gui.deleteLater();app.processEvents()


def test_peft_lokr_initialization_control(monkeypatch,tmp_path):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(TrainingGUI,'_load_presets',lambda self,**kw:None)
    gui=TrainingGUI(preferences=Preferences(tmp_path/'gui.json'))
    try:
        gui._apply(bridge.defaults()|{'dataset.path':'/images','train.model_family':'moss_image','adapter.kind':'lokr'})
        assert gui.editors['adapter.lokr_init'].widget.isEnabled()
        gui.editors['adapter.lokr_init'].set('lycoris');gui._on_edit()
        path=tmp_path/'lokr.toml';bridge.write_toml(path,gui.collect())
        assert load_config(path).adapter.build().init_weights=='lycoris'
        gui.editors['adapter.kind'].set('lora');gui._on_edit()
        assert not gui.editors['adapter.lokr_init'].widget.isEnabled()
    finally:
        gui.close();gui.deleteLater();app.processEvents()
