"""Run the original 161-check gate without touching the user's configs/preferences."""

import os
import shutil
from pathlib import Path


def test_original_gui_parity_gate_in_isolated_workspace(tmp_path, monkeypatch, capsys):
    from trainer.parity import test_gui as gate
    from trainer.gui import app
    from trainer.ui_core.preferences import Preferences

    root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    (tmp_path / "trainer/training").mkdir(parents=True)
    shutil.copyfile(
        root / "trainer/training/train.py", tmp_path / "trainer/training/train.py"
    )
    monkeypatch.setattr(gate, "__file__", str(tmp_path / "trainer/parity/test_gui.py"))
    monkeypatch.setattr(gate, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(app, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(
        app, "Preferences", lambda: Preferences(tmp_path / "prefs/gui.json")
    )
    previous = Path.cwd()
    try:
        result = gate.main()
    finally:
        os.chdir(previous)
    assert result == 0, capsys.readouterr().out
