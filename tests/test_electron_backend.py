"""Desktop contracts exercised with real config loading and controlled CPU workers."""

import copy
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from trainer.ui_core.config import bridge, descriptors, normalize
from trainer.ui_core.preferences import Preferences
from trainer.ui_core.process import CallbackSignal
from trainer.ui_core.session import Session


@pytest.fixture
def session(tmp_path):
    instance = Session(
        root=tmp_path,
        config_dir=tmp_path / "configs",
        preferences=Preferences(tmp_path / "prefs/gui.json"),
        gpus=[dict(index=0, name="Test GPU")],
    )
    instance.flat = normalize(
        bridge.defaults()
        | {
            "dataset.path": str(tmp_path / "images"),
            "train.model_family": "moss_image",
            "train.run_name": "first",
            "train.output_dir": str(tmp_path / "output"),
            "adapter.kind": "lora",
        }
    )
    yield instance
    instance.close()


def source(session, name):
    path = session.root / f"{name}.toml"
    bridge.write_toml(path, session.flat | {"train.run_name": name})
    return path


class ControlledWorker:
    launches = []
    on_run = None
    exit_code = 0

    def __init__(self, launch, cwd, env):
        self.launch, self.env = launch, env
        self.stop_requested = False
        for name in (
            "logSignal",
            "progressSignal",
            "errorSignal",
            "metricsSignal",
            "finishedSignal",
        ):
            setattr(self, name, CallbackSignal())

    def run(self):
        type(self).launches.append(self.launch)
        if type(self).on_run:
            type(self).on_run(self)
        self.finishedSignal.emit(-1 if self.stop_requested else type(self).exit_code)

    def stop(self):
        self.stop_requested = True


@pytest.fixture
def worker(session):
    ControlledWorker.launches = []
    ControlledWorker.on_run = None
    ControlledWorker.exit_code = 0
    session.worker_factory = ControlledWorker
    return ControlledWorker


def finish(session):
    session.thread.join(timeout=10)
    assert not session.thread.is_alive()
    assert not session.busy


def test_complete_console_history_survives_bounded_live_tail(session):
    from trainer.gui.widgets import CompressedLogBuffer as QtBuffer
    from trainer.ui_core.console import CompressedLogBuffer

    assert QtBuffer is CompressedLogBuffer
    for i in range(11000):
        session.log(f"Training line {i} 日本語")
    session.log("50%", progress=True)
    session.log("75%", progress=True)
    assert len(session.logs) == 10000
    assert session.console_read(0, 2)["lines"] == [
        "Training line 0 日本語",
        "Training line 1 日本語",
    ]
    assert session.console_read(count=2)["lines"] == [
        "Training line 10999 日本語",
        "75%",
    ]
    assert session.console_read()["total"] == 11001
    assert len(session.console_read(0, 100000)["lines"]) == 1200
    text = session.console_text()
    assert text.startswith("Training line 0 日本語\n")
    assert text.endswith("Training line 10999 日本語\n75%")
    assert "50%" not in text
    stored, original, ratio = session.console.memory_summary()
    assert stored < original and ratio < 1
    session.console_clear()
    assert session.console_read()["lines"] == []
    assert session.console_text() == ""
    assert not session.logs


def test_metric_compaction_preserves_timeline_and_extrema():
    from trainer.ui_core.metrics import compact_records

    records = [
        dict(step=i, loss=float(i % 100), lr=0.001, peak=[8.0, 9.0])
        for i in range(61000)
    ]
    records[125]["loss"] = 100000.0
    records[4125]["peak"] = [99.0, 9.0]
    records[32125]["peak"] = [8.0, 99.0]
    result = compact_records(records)
    assert len(result) < 60000
    assert result[0] is records[0] and result[-1] is records[-1]
    for index in (125, 4125, 32125):
        assert records[index] in result
    assert [r["step"] for r in result] == sorted(r["step"] for r in result)


def test_headless_coverage_and_renderer_kinds():
    result = subprocess.run(
        [sys.executable, "-m", "trainer.electron_backend", "--check"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert json.loads(result.stdout) == dict(ok=True, fields=197, qtImported=False)
    definition = descriptors()
    assert bridge.missing_ui_keys(definition["fields"]) == ([], [])
    from trainer.gui.schema import layout_keys

    assert set(layout_keys()) == set(definition["fields"])
    assert len(layout_keys()) == len(set(layout_keys()))
    renderer = (
        Path(__file__).resolve().parents[1] / "electron-ui/src/Fields.tsx"
    ).read_text("utf-8")
    for field in definition["fields"].values():
        assert repr(field["kind"]) in renderer
    assert definition["curriculumPreset"][0]["at"] == 0
    assert len(definition["curriculumPreset"]) == 9


def test_normalization_stages_optimizer_and_quantization(session):
    state = session.edit(
        {
            "resolution_schedule.enabled": True,
            "dataset.source": "latents",
            "dataset.resolutions": [256, 384, 512, 640, 768, 1024],
        }
    )
    flat = state["flat"]
    assert abs(sum(s["share"] for s in flat["resolution_schedule.stages"]) - 1) < 1e-10
    assert flat["train.batch_size"] == {t: 1 for t in [256, 384, 512, 640, 768, 1024]}
    assert flat["train.max_steps"] == 560
    assert state["validation"]["valid"], state["validation"]["error"]
    before = copy.deepcopy(flat)
    session.edit({"optimizer.kind": "adamw", "optimizer.lr": 0.0002})
    assert session.flat["optimizer.lr"] == 0.0002
    session.edit({"flow.timestep_sample_method": "moss", "flow.shift": 5})
    assert session.flat["flow.shift"] == 1
    assert before["resolution_schedule.stages"] == flat["resolution_schedule.stages"]


def test_save_external_copy_rename_and_collision(session):
    external = source(session, "external")
    original = external.read_bytes()
    session.open(external)
    first = Path(session.save()["path"])
    assert first == session.config_dir / "external.toml"
    assert external.read_bytes() == original
    session.edit({"train.run_name": "copy"})
    second = Path(session.save()["path"])
    assert first.exists() and second.exists()
    session.edit({"train.run_name": "external"})
    assert session.save()["confirm"] == "overwrite"
    assert session.save(overwrite=True)["path"] == str(first)


def test_unknown_values_and_cli_only_settings_roundtrip(session):
    session.flat["adapter.init"] = "mid"
    session.flat["quant.skip_policy"] = "future-policy"
    path = session.save()["path"]
    session.open(path)
    assert session.flat["adapter.init"] == "mid"
    assert session.flat["quant.skip_policy"] == "future-policy"
    assert not session.state()["validation"]["valid"]


def test_subset_edit_disables_single_folder(session):
    result = session.edit(
        {"dataset.subsets": [dict(path="画像/data", num_repeats=2, texture=False)]}
    )
    assert result["flat"]["dataset.path"] is None
    assert result["validation"]["enabled"]["dataset.path"] is False
    assert result["validation"]["enabled"]["dataset.num_repeats"] is False
    path = session.save()["path"]
    assert bridge.read_toml(path)["dataset"]["subsets"][0]["path"] == "画像/data"


def test_snapshot_survives_source_edit_and_delete(session, worker):
    path = source(session, "queued")
    session.add_queue([path])
    saved = Path(session.entries[0].snapshot).read_bytes()
    path.write_text("broken", encoding="utf-8")
    path.unlink()
    session.start_queue()
    finish(session)
    assert session.entries[0].status == "Completed"
    assert Path(session.entries[0].snapshot).read_bytes() == saved
    assert any(
        "trainer.tools.cache_latents" in launch.argv for launch in worker.launches
    )
    assert "trainer.training.train" in worker.launches[-1].argv


def test_queue_duplicates_release_reservation_after_remove(session, worker):
    a = source(session, "a")
    b = source(session, "b")
    session.add_queue([a, b])
    assert len(session.entries) == 2
    # A copy with a different filename and the same output must be rejected.
    duplicate = session.root / "duplicate.toml"
    duplicate.write_bytes(a.read_bytes())
    assert session.add_queue([duplicate])["errors"]
    session.queue_edit("remove", [session.entries[0].id])
    assert not session.add_queue([duplicate])["errors"]
    session.start_queue()
    finish(session)
    assert all(e.status == "Completed" for e in session.entries)


@pytest.mark.parametrize("keep_going", [False, True])
def test_failed_cache_skips_its_training(session, worker, keep_going):
    session.add_queue([source(session, "a"), source(session, "b")])
    calls = []

    def outcome(current):
        calls.append(current.launch)
        worker.exit_code = (
            3
            if "trainer.tools.cache_latents" in current.launch.argv
            and len([c for c in calls if "trainer.tools.cache_latents" in c.argv]) == 1
            else 0
        )

    worker.on_run = outcome
    session.start_queue(keep_going)
    finish(session)
    assert session.entries[0].status == "Failed"
    assert session.entries[1].status == ("Completed" if keep_going else "Skipped")
    assert len([l for l in calls if "trainer.training.train" in l.argv]) == (
        1 if keep_going else 0
    )


def test_stop_wins_over_continue_and_live_pending_edits(session, worker):
    a, b, c = [source(session, name) for name in ("a", "b", "c")]
    session.add_queue([a, b])
    entered, release = threading.Event(), threading.Event()
    worker.on_run = lambda current: (entered.set(), release.wait(5))
    session.start_queue(True)
    assert entered.wait(5)
    active = session.entries[0]
    session.add_queue([c])
    session.queue_edit("move", [session.entries[-1].id], -1)
    assert session.entries[0] is active
    session.queue_edit("remove", [active.id])
    assert active in session.entries
    session.stop()
    release.set()
    finish(session)
    assert active.status == "Stopped"
    assert all(e.status == "Skipped" for e in session.entries[1:])
    assert len(worker.launches) == 1


def test_snapshot_tamper_between_steps_blocks_training(session, worker):
    session.add_queue([source(session, "a")])
    entry = session.entries[0]

    def tamper(current):
        if "trainer.tools.cache_latents" in current.launch.argv:
            Path(entry.snapshot).write_text("tampered")

    worker.on_run = tamper
    session.start_queue()
    finish(session)
    assert entry.status == "Failed"
    assert "modified" in entry.error
    assert not any("trainer.training.train" in l.argv for l in worker.launches)


def test_retry_uses_snapshot_and_metrics_freeze_before_next_reset(session, worker):
    events = []
    session.emit = events.append
    session.add_queue([source(session, "a"), source(session, "b")])

    def output(current):
        if "trainer.training.train" in current.launch.argv:
            current.metricsSignal.emit(
                "e1 step 1/1 loss 0.12 lr 1e-4 1.8s/it peak_alloc 8.1GiB"
            )

    worker.on_run = output
    session.start_queue()
    finish(session)
    snapshots = [e for e in events if e["event"] == "metrics_snapshot"]
    assert len(snapshots) == 2
    assert snapshots[0]["metrics"]["records"][0]["loss"] == 0.12
    assert snapshots[0]["path"] != snapshots[1]["path"]
    assert all(Path(e["path"]).name == "metrics-final.png" for e in snapshots)
    session.entries[0].status = "Failed"
    original = Path(session.entries[0].snapshot).read_bytes()
    session.queue_edit("retry", [session.entries[0].id])
    assert session.entries[0].status == "Pending"
    assert Path(session.entries[0].snapshot).read_bytes() == original


def test_autosave_off_and_single_start_confirmation(session, worker):
    session.preferences.set("autosave", False)
    session.add_queue([source(session, "queued")])
    assert session.start()["confirm"] == "queue"
    assert not worker.launches
    session.start(confirmed=True)
    finish(session)
    assert not list(session.config_dir.glob("*.toml"))
    assert session.entries[0].status == "Pending"
    assert "trainer.training.train" in worker.launches[-1].argv


def test_metrics_pause_clear_keep_wall_clock(session):
    session.metrics["startedAt"] = 1000
    line = "e1 step 1/2 loss 0.12 lr 1e-4 1.8s/it peak_alloc 8.1GiB"
    session._metric_line(line, session.flat, None, True)
    assert len(session.metrics["records"]) == 1
    session.metrics_control("pause")
    session._metric_line(line, session.flat, None, True)
    assert len(session.metrics["records"]) == 1
    session.metrics_control("clear")
    assert not session.metrics["records"] and session.metrics["startedAt"] == 1000


def test_preferences_keep_legacy_values_and_unicode(session):
    session.set_preferences(
        dict(
            autosave=False,
            show_experimental=True,
            config_dir=str(session.root / "設定"),
        )
    )
    reread = Preferences(session.preferences.path)
    assert reread.values["show_experimental"] is True
    assert not reread.values["autosave"]
    assert session.config_dir.name == "設定"


def test_rpc_unknown_method_rejected(session):
    from trainer.electron_backend.__main__ import Server

    server = Server(session, lambda message: None)
    with pytest.raises(ValueError, match="Unknown desktop operation"):
        server.handle(dict(method="exec", params={"command": "anything"}))


def test_invalid_toml_is_snapshotted_then_skipped_when_continuing(session, worker):
    invalid = session.root / "invalid.toml"
    invalid.write_text("[train\nthis is invalid", encoding="utf-8")
    session.add_queue([invalid, source(session, "valid")])
    assert len(session.entries) == 2
    assert Path(session.entries[0].snapshot).read_bytes() == invalid.read_bytes()
    session.start_queue(True)
    finish(session)
    assert [e.status for e in session.entries] == ["Failed", "Completed"]


def test_stop_after_completion_keeps_completed_status(session, worker):
    session.add_queue([source(session, "a"), source(session, "b")])

    def stop_after_snapshot(event):
        if event["event"] == "metrics_snapshot":
            session.stop()

    session.emit = stop_after_snapshot
    session.start_queue(True)
    finish(session)
    assert [e.status for e in session.entries] == ["Completed", "Skipped"]


def test_latent_preview_changes_distribution_without_changing_config(session):
    session.edit(
        {
            "flow.timestep_sample_method": "logit_normal",
            "flow.flux_shift": True,
            "flow.shift": None,
        }
    )
    before = copy.deepcopy(session.flat)
    first = session.set_preview(32, 32)["validation"]["distribution"]
    second = session.set_preview(128, 128)["validation"]["distribution"]
    assert first != second
    assert session.flat == before
    assert session.state()["previewGrid"] == [128, 128]
    with pytest.raises(ValueError):
        session.set_preview(0, 64)


def test_wire_preserves_sets_and_invalid_floats_as_editable_values():
    from trainer.electron_backend.__main__ import wire_value

    assert wire_value({"tags": {"b", "a"}, "invalid": float("nan")}) == {
        "tags": ["a", "b"],
        "invalid": "nan",
    }


def test_qt_numeric_display_does_not_rewrite_untouched_config_values():
    from PySide6 import QtWidgets
    from trainer.gui.fields import FloatEditor

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    editor = FloatEditor(0, 1, decimals=12)
    editor.set(1 / 9)
    assert editor.get() == 1 / 9
    editor.widget.setValue(0.2)
    assert editor.get() == 0.2
    editor.widget.deleteLater()
    app.processEvents()


def test_all_gpu_selection_respects_inherited_device_environment(session, monkeypatch):
    from trainer.ui_core.process import training_env

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert session.gpu_arg() == ""
    assert training_env(session.gpu_arg())["CUDA_VISIBLE_DEVICES"] == "3"
    session.gpus.append(dict(index=1, name="Another GPU"))
    assert session.gpu_arg() == "0"


def test_cache_paths_ignore_empty_and_whitespace_rows(session):
    assert session._paths({"dataset.path": "   "}) == []
    assert session._paths(
        {
            "dataset.subsets": [
                {"path": "  "},
                {"path": " /data/a "},
                {"path": "/data/a"},
            ]
        }
    ) == ["/data/a"]
