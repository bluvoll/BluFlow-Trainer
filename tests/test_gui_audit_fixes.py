"""Regression tests for GUI & TOML QoL fixes (Workstreams A–D)."""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from trainer.gui import bridge
from trainer.training.config import load_config
from trainer.ui_core.config import descriptors, evaluate, normalize
from trainer.ui_core.preferences import Preferences
from trainer.ui_core.process import cache_launch, train_launch
from trainer.ui_core.rules import _RULES
from trainer.ui_core.session import Session


def test_flow_shift_cleared_serializes_identity_one(tmp_path):
    flat = bridge.defaults() | {
        "dataset.path": "/images",
        "train.model_family": "moss_image",
        "flow.timestep_sample_method": "logit_normal",
        "flow.cabal_shift": False,
        "flow.flux_shift": False,
        "flow.shift": None,
    }
    p = tmp_path / "no_shift.toml"
    bridge.write_toml(p, flat)
    cfg = load_config(p)
    assert cfg.flow.shift == 1.0
    ev = evaluate(flat)
    assert ev["distribution"]["shift"] == 1.0


def test_save_every_epochs_cleared_roundtrips_zero_to_none(tmp_path):
    flat = bridge.defaults() | {
        "dataset.path": "/images",
        "train.model_family": "moss_image",
        "train.save_every_epochs": None,
    }
    p = tmp_path / "only_final.toml"
    bridge.write_toml(p, flat)
    cfg = load_config(p)
    assert cfg.train.save_every_epochs == 0
    reloaded = bridge.flatten(bridge.read_toml(p))
    assert reloaded["train.save_every_epochs"] is None


def test_flux_shift_toml_clears_default_shift_for_previews(tmp_path):
    raw = {
        "dataset": {"path": "/images"},
        "train": {"model_family": "moss_image"},
        "flow": {"flux_shift": True},
    }
    merged = normalize(bridge.defaults() | bridge.flatten(raw))
    assert merged["flow.flux_shift"] is True
    assert merged["flow.shift"] is None
    ev = evaluate(merged)
    assert ev["timestepError"] == ""
    assert ev["distribution"] is not None


def test_disabled_resolution_schedule_pruned_and_rebuilt(tmp_path):
    flat = bridge.defaults() | {
        "dataset.path": "/images",
        "train.model_family": "moss_image",
        "dataset.resolutions": [512, 1024, 1280],
        "resolution_schedule.enabled": False,
        "resolution_schedule.unit": "stage_epochs",
        "resolution_schedule.stages": [{"resolution": 512, "epochs": 2, "lr_mul": 1.0}],
    }
    notes = bridge.advisories(flat)
    assert any("resolution_schedule.stages" in msg for _, msg in notes)
    pruned = bridge.prune_defaults(flat)
    assert "resolution_schedule.stages" not in pruned
    assert "resolution_schedule.unit" not in pruned

    # Re-enabling with mismatched stage resolutions rebuilds stages for dataset.resolutions.
    enabled = normalize(flat | {"resolution_schedule.enabled": True}, flat)
    assert [s["resolution"] for s in enabled["resolution_schedule.stages"]] == [512, 1024, 1280]


def test_disabled_dependent_controls_reset_and_min_source_area_enabled():
    base = bridge.defaults() | {
        "dataset.path": "/images",
        "train.model_family": "moss_image",
        "adapter.kind": "lora",
        "train.resume_from": "/ckpt",
        "train.resume_mode": "weights",
        "train.cache_text_embeddings": True,
        "train.caption_variations": 3,
        "component_lr.adaln": 1e-4,
    }
    # Single-tier mode still allows dataset.min_source_area.
    assert _RULES["dataset.min_source_area"](base | {"dataset.resolutions": []}) is True

    prev = dict(base)
    updated = normalize(
        base | {"train.resume_from": None, "train.cache_text_embeddings": False},
        prev,
    )
    assert updated["train.resume_mode"] == "exact"
    assert updated["train.caption_variations"] == 0
    assert updated["component_lr.adaln"] is None
    pruned = bridge.prune_defaults(updated)
    assert "component_lr.adaln" not in pruned


def test_flatten_handles_curriculum_asdict_and_protected_tags_file(tmp_path):
    p = tmp_path / "cfg.toml"
    tags_file = tmp_path / "protected.txt"
    tags_file.write_text("1girl\n", encoding="utf-8")
    p.write_text(
        f"""
[train]
model_family = "moss_image"
[dataset]
path = "/images"
source = "encode"
[dataset.caption]
protected_tags_file = "{tags_file}"
[[curriculum]]
at = 0.0
t_range = [0.0, 1.0]
mode = "texture"
""",
        encoding="utf-8",
    )
    raw = bridge.read_toml(p)
    flat_raw = bridge.flatten(raw)
    assert flat_raw["dataset.caption.protected_tags_file"] == str(tags_file)
    assert isinstance(flat_raw["curriculum"], list) and len(flat_raw["curriculum"]) == 1

    cfg = load_config(p)
    flat_dc = bridge.flatten(asdict(cfg))
    assert isinstance(flat_dc["curriculum"], list) and len(flat_dc["curriculum"]) == 1
    assert flat_dc["curriculum"][0]["mode"] == "texture"

    # protected_tags = [] matches empty set default and is pruned cleanly.
    out = bridge.prune_defaults(bridge.defaults() | {"dataset.path": "/images", "dataset.caption.protected_tags": []})
    assert "dataset.caption.protected_tags" not in out


def test_float_editor_specs_steps_and_decimals():
    fields = descriptors()["fields"]
    assert fields["perceptual.weight"]["kwargs"] == {"step": 0.01, "decimals": 6}
    assert fields["perceptual.max_t"]["kwargs"] == {"step": 0.01, "decimals": 3}
    assert fields["self_flow.decay"]["kwargs"] == {"step": 0.001, "decimals": 6}
    assert fields["self_flow.weight"]["kwargs"] == {"step": 0.01, "decimals": 4}
    assert fields["rti.start_keep"]["kwargs"] == {"step": 0.01, "decimals": 4}
    assert fields["rti.target_keep"]["kwargs"] == {"step": 0.01, "decimals": 4}
    assert fields["dataset.resolution"]["args"] == [128, 8192, 64]


def test_train_launch_and_cache_jobs_ddp_and_source_skip(tmp_path, monkeypatch):
    launch1 = train_launch("cfg.toml", num_processes=1)
    assert "--num_machines" in launch1.argv and "--mixed_precision" in launch1.argv
    assert "--gpu_ids" in launch1.argv and "--multi_gpu" not in launch1.argv

    launch2 = train_launch("cfg.toml", num_processes=2)
    assert "--multi_gpu" in launch2.argv and "--num_processes" in launch2.argv

    c_launch = cache_launch("/images", "moss-image", [512, 1024],
                            min_bucket_reso=256, max_bucket_reso=4096,
                            bucket_reso_steps=64, upscale=False, multires_training=True,
                            gpus="", devices="0,1")
    assert "--devices" in c_launch.argv and "0,1" in c_launch.argv
    assert "CUDA_VISIBLE_DEVICES" not in c_launch.env

    s = Session(
        root=tmp_path,
        config_dir=tmp_path / "configs",
        preferences=Preferences(tmp_path / "prefs.json"),
        gpus=[dict(index=0, name="GPU 0"), dict(index=1, name="GPU 1")],
    )
    try:
        s.flat = normalize(
            bridge.defaults()
            | {
                "dataset.path": str(tmp_path / "images"),
                "dataset.source": "latents",
                "train.model_family": "moss_image",
                "train.output_dir": str(tmp_path / "out"),
                "adapter.kind": "lora",
            }
        )
        assert "--devices" in s.cache_jobs(s.flat)[0].launch.argv
        captured = []
        monkeypatch.setattr(s, "_launch_thread", lambda op: op())
        monkeypatch.setattr(s, "_sequence", lambda jobs, *a: captured.extend(jobs))
        s.start("train")
        assert not any("trainer.tools.cache_latents" in j.launch.argv for j in captured)
        assert any("trainer.training.train" in j.launch.argv for j in captured)

        captured.clear()
        s.start("cache")
        assert any("trainer.tools.cache_latents" in j.launch.argv for j in captured)
    finally:
        s.close()


def test_session_preserves_single_path_across_subset_toggle(tmp_path):
    s = Session(
        root=tmp_path,
        config_dir=tmp_path / "configs",
        preferences=Preferences(tmp_path / "prefs.json"),
        gpus=[dict(index=0, name="GPU 0")],
    )
    try:
        s.edit({"dataset.path": "/my/dataset"})
        res1 = s.edit({"dataset.subsets": [{"path": "/sub/a", "num_repeats": 1, "texture": False}]})
        assert res1["flat"]["dataset.path"] is None
        res2 = s.edit({"dataset.subsets": []})
        assert res2["flat"]["dataset.path"] == "/my/dataset"
    finally:
        s.close()


def test_qt_editors_typos_curriculum_insert_and_gpu_refresh(monkeypatch, tmp_path):
    from trainer.gui.app import TrainingGUI, detect_gpus
    from trainer.gui.fields import (
        BucketMapEditor,
        CanvasListEditor,
        CurriculumEditor,
        NumListEditor,
        OptIntEditor,
        PathEditor,
        ResolutionStagesEditor,
        SciEditor,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    pe = PathEditor()
    pe.widget.findChild(QtWidgets.QLineEdit).setText("   ")
    assert pe.get() is None
    pe.widget.deleteLater()

    pe_req = PathEditor("folder", optional=False)
    pe_req.widget.findChild(QtWidgets.QLineEdit).setText("   ")
    assert pe_req.get() == ""
    pe_req.widget.deleteLater()

    oie = OptIntEditor()
    oie.widget.setText("bad")
    assert oie.get() == "bad"
    oie.widget.deleteLater()

    se = SciEditor()
    se.widget.setText("1e-x")
    assert se.get() == "1e-x"
    se.widget.deleteLater()

    nle = NumListEditor()
    nle.widget.setText("512, oops")
    assert nle.get() == "512, oops"
    nle.widget.deleteLater()

    bme = BucketMapEditor()
    bme.widget.setText("512=oops")
    assert bme.get() == "512=oops"
    bme.widget.setText("")
    assert bme.get() == ""
    bme.widget.deleteLater()

    cle = CanvasListEditor()
    cle.widget.setPlainText("1024x1024, bad")
    assert cle.get() == "1024x1024, bad"
    cle.widget.deleteLater()

    ce = CurriculumEditor()
    ce._add_row()
    ce._add_row()
    assert ce.enable.isChecked()
    assert len(ce.get()) == 2
    ce.widget.deleteLater()

    rse = ResolutionStagesEditor()
    rse.set_unit("stage_epochs")
    assert rse.explicit_epochs is True
    assert rse.table.horizontalHeaderItem(1).text() == "Epochs"
    rse.widget.deleteLater()

    monkeypatch.setattr(TrainingGUI, "_load_presets", lambda self: None)
    monkeypatch.setattr("trainer.gui.app.detect_gpus", lambda: [(0, "GPU 0"), (1, "GPU 1")])
    gui = TrainingGUI(preferences=Preferences(tmp_path / "gui.json"))
    try:
        gui._apply(bridge.defaults() | {
            "dataset.path": "/images",
            "train.model_family": "moss_image",
            "curriculum": [{"at": 0.0, "t_range": [0.0, 1.0], "mode": "texture"}],
            "dataset.source": "encode",
        })
        assert not gui.start_btn.isEnabled()
        gui.gpu_boxes[1].setChecked(False)
        assert gui.num_processes() == 1
        assert gui.start_btn.isEnabled()
    finally:
        gui.close()
        gui.deleteLater()
        app.processEvents()
