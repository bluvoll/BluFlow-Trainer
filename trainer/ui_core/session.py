"""Headless desktop session: configs, immutable queues and supervised workers.

The UI can disappear without corrupting a config. Jobs never read mutable editor state.
"""

from __future__ import annotations
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

from .config import bridge, descriptors, normalize, evaluate
from .metrics import parse_step, REPORT_RE, compact_records
from .paths import PROJECT_ROOT, safe_stem
from .preferences import Preferences
from .process import (
    Job,
    Launch,
    ProcessWorker,
    train_launch,
    cache_launch,
    audit_launch,
    calibration_launch,
    toolchain_jobs,
    training_env,
)
from .queue import QueueEntry, snapshot_file
from .console import CompressedLogBuffer


def detect_gpus():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return [
            dict(index=int(line.partition(",")[0]), name=line.partition(",")[2].strip())
            for line in result.stdout.splitlines()
            if "," in line
        ] or [dict(index=0, name="GPU 0")]
    except (OSError, ValueError, subprocess.SubprocessError):
        return [dict(index=0, name="GPU 0")]


class Session:
    def __init__(
        self,
        emit=None,
        *,
        root=PROJECT_ROOT,
        preferences=None,
        config_dir=None,
        gpus=None,
        worker_factory=ProcessWorker,
    ):
        self.root = Path(root).resolve()
        self.preferences = preferences if preferences is not None else Preferences()
        self.config_dir = (
            Path(
                config_dir
                or self.preferences.values.get("config_dir")
                or self.root / "configs"
            )
            .expanduser()
            .resolve()
        )
        self.emit = emit or (lambda event: None)
        self.lock = threading.RLock()
        self.worker_factory = worker_factory
        self.gpus = gpus if gpus is not None else detect_gpus()
        self.selected = [
            g["index"]
            for g in self.gpus[: 1 if sys.platform == "win32" else len(self.gpus)]
        ]
        self.flat = normalize(bridge.defaults())
        self.path = None
        self.external = []
        self.storage = tempfile.TemporaryDirectory(prefix="bluflow-desktop-")
        self.entries: list[QueueEntry] = []
        self.next_id = 0
        self.reservations = {}
        self.busy = False
        self.queue_active = False
        self.continue_errors = False
        self.cancelled = threading.Event()
        self.worker = None
        self.thread = None
        self.active = None
        self.logs = deque(maxlen=10000)
        self.console = CompressedLogBuffer(block_size=128, compression_level=6)
        self.last_progress = False
        self.tail = ""
        self.metrics_paused = False
        self.metrics = dict(
            records=[], latest={}, reports=[], startedAt=None, elapsed=None
        )
        self.snapshot_sent = False
        self.preview_grid = (64, 64)
        presets = self.presets()
        if presets:
            try:
                self.open(presets[0]["path"])
            except (ValueError, OSError) as exc:
                self.log(f"Could not load preset: {exc}")

    def event(self, name, **data):
        self.emit(dict(event=name, **data))

    def log(self, text, progress=False):
        with self.lock:
            record = dict(text=str(text), progress=progress)
            self.console.append(text, replace_last=progress and self.last_progress)
            self.last_progress = progress
            if progress and self.logs and self.logs[-1]["progress"]:
                self.logs.pop()
            self.logs.append(record)
            self.tail = (self.tail + str(text) + "\n")[-6000:]
        self.event("log", **record, total=self.console.line_count)

    def console_read(self, start=None, count=900):
        with self.lock:
            total = self.console.line_count
            count = max(0, min(1200, int(count)))
            start = max(
                0,
                min(
                    max(0, total - count),
                    int(start) if start is not None else max(0, total - count),
                ),
            )
            return dict(
                lines=self.console.get_lines(start, count),
                start=start,
                total=total,
                memory=self.console.memory_summary(),
            )

    def console_text(self):
        with self.lock:
            return self.console.get_all_text()

    def console_clear(self):
        with self.lock:
            self.console.clear()
            self.logs.clear()
            self.last_progress = False
        self.event("console_clear")
        return True

    def presets(self):
        paths = list(self.config_dir.glob("*.toml")) if self.config_dir.is_dir() else []
        paths += [p for p in self.external if p.is_file() and p not in paths]
        if self.path and self.path.is_file() and self.path not in paths:
            paths.append(self.path)
        return [
            dict(path=str(p), name=p.stem)
            for p in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
        ]

    def state(self):
        with self.lock:
            return dict(
                flat=copy.deepcopy(self.flat),
                path=str(self.path) if self.path else None,
                configDir=str(self.config_dir),
                preferences=dict(self.preferences.values),
                presets=self.presets(),
                gpus=self.gpus,
                selectedGpus=list(self.selected),
                multiGpu=sys.platform != "win32",
                platform=sys.platform,
                busy=self.busy,
                queueActive=self.queue_active,
                active=self.active,
                queue=[asdict(e) for e in self.entries],
                continueErrors=self.continue_errors,
                validation=evaluate(
                    self.flat,
                    len(self.selected),
                    self.preferences.values.get("show_experimental", False),
                    *self.preview_grid,
                ),
                previewGrid=list(self.preview_grid),
            )

    def publish(self):
        self.event("state", state=self.state())

    def bootstrap(self):
        return dict(
            **descriptors(),
            state=self.state(),
            metrics=copy.deepcopy(self.metrics),
            logs=list(self.logs),
            consoleTotal=self.console.line_count,
        )

    def open(self, path):
        path = Path(path).expanduser().resolve()
        flat = bridge.flatten(bridge.read_toml(path))
        values = bridge.defaults(flat.get("optimizer.kind", "adamw")) | flat
        if values.get("adapter.kind") == "lycoris_lora" and "train.compile" not in flat:
            values["train.compile"] = "default"
        with self.lock:
            self.flat, self.path = normalize(values), path
            if path not in self.external:
                self.external.append(path)
        self.log(f"Loaded {path}")
        return self.state()

    def edit(self, patch):
        if not isinstance(patch, dict) or any(
            k not in self.flat and k not in descriptors()["fields"] for k in patch
        ):
            raise ValueError("Unknown configuration field")
        with self.lock:
            self.flat = normalize(self.flat | patch, self.flat)
        return self.state()

    def save(self, destination=None, overwrite=False):
        with self.lock:
            flat = copy.deepcopy(self.flat)
            target = (
                Path(destination).expanduser().resolve()
                if destination
                else self.config_dir / f"{safe_stem(flat.get('train.run_name'))}.toml"
            )
            if target.exists() and target != self.path and not overwrite:
                return dict(
                    confirm="overwrite",
                    path=str(target),
                    message=f"{target.name} already exists. Overwrite this config? Runs sharing this name also share checkpoint output.",
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            bridge.write_toml(target, flat)
            self.path = target
            if target not in self.external:
                self.external.append(target)
        self.log(f"Saved {target}")
        return dict(path=str(target), state=self.state())

    def set_preferences(self, patch):
        allowed = {
            "config_dir",
            "autosave",
            "show_experimental",
            "windows_allocator",
            "collapsed_sections",
        }
        if not isinstance(patch, dict) or set(patch) - allowed:
            raise ValueError("Unknown preference")
        if "windows_allocator" in patch and patch["windows_allocator"] not in (
            "inherit",
            "native",
            "cudaMallocAsync",
        ):
            raise ValueError("Unknown CUDA allocator")
        if "config_dir" in patch:
            directory = (
                Path(patch["config_dir"]).expanduser()
                if patch["config_dir"]
                else self.root / "configs"
            )
            directory.mkdir(parents=True, exist_ok=True)
            self.config_dir = directory.resolve()
        for key, value in patch.items():
            self.preferences.set(key, value)
        return self.state()

    def select_gpus(self, selected):
        if self.busy:
            raise ValueError("GPU selection is locked while a job runs")
        selected = sorted(set(selected))
        if (
            not selected
            or set(selected) - {g["index"] for g in self.gpus}
            or (sys.platform == "win32" and len(selected) != 1)
        ):
            raise ValueError("Select one GPU on Windows, or at least one GPU on Linux")
        self.selected = selected
        return self.state()

    def set_preview(self, height, width):
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 512
            for value in (height, width)
        ):
            raise ValueError(
                "Preview latent dimensions must be integers between 1 and 512"
            )
        self.preview_grid = (height, width)
        return self.state()

    def check(self, flat):
        valid = evaluate(flat, len(self.selected))
        if not valid["valid"]:
            raise ValueError(valid["error"])
        errors = [m for level, m in valid["advisories"] if level == "error"]
        if errors:
            raise ValueError("; ".join(errors))

    def _destination(self, flat):
        return (
            self.root
            / (flat.get("train.output_dir") or "output")
            / (flat.get("train.run_name") or "mageflow")
        ).resolve()

    def add_queue(self, paths):
        errors = []
        with self.lock:
            for path in paths:
                try:
                    source = Path(path).resolve()
                    data = source.read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    if any(
                        e.source == str(source) and e.digest == digest
                        for e in self.entries
                    ):
                        continue
                    # Malformed files are still snapshotted and diagnosed at preflight.
                    try:
                        flat = bridge.flatten(bridge.read_toml(source))
                    except (ValueError, TypeError):
                        flat = None
                    if flat is not None:
                        destination = self._destination(flat)
                        for existing in self.entries:
                            try:
                                existing_flat = bridge.flatten(
                                    bridge.read_toml(existing.snapshot)
                                )
                            except (OSError, ValueError, TypeError):
                                continue
                            if self._destination(existing_flat) == destination:
                                raise ValueError(
                                    f"Duplicate output directory/run_name: {destination}"
                                )
                    entry = snapshot_file(source, self.storage.name, self.next_id, data)
                    entry.source, entry.snapshot = (
                        str(entry.source),
                        str(entry.snapshot),
                    )
                    self.entries.append(entry)
                    self.next_id += 1
                except (OSError, ValueError, TypeError) as exc:
                    errors.append(f"{path}: {exc}")
        for error in errors:
            self.log(error)
        return dict(state=self.state(), errors=errors)

    def queue_edit(self, operation, ids=(), direction=0):
        with self.lock:
            selected = set(ids)
            if operation in ("remove", "clear"):
                remove = [
                    e
                    for e in self.entries
                    if (
                        (operation == "clear" and e.status == "Pending")
                        or e.id in selected
                    )
                    and e.status not in ("Caching", "Training")
                    and (not self.busy or e.status == "Pending")
                ]
                for e in remove:
                    self.entries.remove(e)
                    Path(e.snapshot).unlink(missing_ok=True)
                    self.reservations = {
                        p: owner
                        for p, owner in self.reservations.items()
                        if owner != e.id
                    }
            elif operation == "retry":
                for e in self.entries:
                    if e.id in selected and e.status in (
                        "Failed",
                        "Stopped",
                        "Skipped",
                    ):
                        e.status, e.error = "Pending", ""
            elif operation == "move":
                if direction not in (-1, 1):
                    raise ValueError("Move direction must be -1 or 1")
                indices = (
                    range(len(self.entries))
                    if direction < 0
                    else reversed(range(len(self.entries)))
                )
                for index in indices:
                    at = index + direction
                    if (
                        0 <= at < len(self.entries)
                        and self.entries[index].id in selected
                        and self.entries[at].id not in selected
                        and self.entries[index].status
                        == self.entries[at].status
                        == "Pending"
                    ):
                        self.entries[index], self.entries[at] = (
                            self.entries[at],
                            self.entries[index],
                        )
            else:
                raise ValueError("Unknown queue operation")
        return self.state()

    def _paths(self, flat):
        return bridge.dataset_paths(flat)

    def gpu_arg(self):
        return (
            ""
            if len(self.selected) == len(self.gpus)
            else ",".join(map(str, self.selected))
        )

    def cache_jobs(self, flat, dry=False):
        return [
            Job(
                cache_launch(
                    p,
                    flat.get("train.model_path") or "",
                    flat.get("dataset.resolutions")
                    or [flat.get("dataset.resolution") or 1024],
                    min_bucket_reso=flat.get("dataset.min_bucket_reso") or 256,
                    max_bucket_reso=flat.get("dataset.max_bucket_reso") or 4096,
                    bucket_reso_steps=flat.get("dataset.bucket_reso_steps") or 64,
                    upscale=not flat.get("dataset.bucket_no_upscale", True),
                    multires_training=bool(flat.get("dataset.multires_training")),
                    gpus=self.gpu_arg(),
                    vae_path=flat.get("train.vae_path"),
                    flux2_vae=bool(flat.get("train.flux2_vae")),
                    model_family=flat.get("train.model_family") or "auto",
                    cache_validation=flat.get("dataset.cache_validation") or "warn",
                    dry_run=dry,
                )
            )
            for p in self._paths(flat)
        ]

    def _prepare_entry(self, entry):
        if (
            hashlib.sha256(Path(entry.snapshot).read_bytes()).hexdigest()
            != entry.digest
        ):
            raise ValueError(
                "Queued snapshot was modified; remove it and add the config again."
            )
        from ..training.config import load_config

        cfg = load_config(entry.snapshot)
        flat = bridge.flatten(asdict(cfg))
        self.check(flat)
        destination = self._destination(flat)
        owner = self.reservations.get(destination)
        if (
            owner is not None
            and owner != entry.id
            and any(e.id == owner for e in self.entries)
        ):
            raise ValueError(f"Duplicate output directory/run_name: {destination}")
        self.reservations[destination] = entry.id
        return flat, toolchain_jobs(entry.snapshot) + self.cache_jobs(flat) + [
            Job(train_launch(entry.snapshot, len(self.selected)), training=True)
        ]

    def start(self, action="train", confirmed=False, overwrite=False):
        if self.busy:
            raise ValueError("A job is already running")
        flat = copy.deepcopy(self.flat)
        if action == "train":
            self.check(flat)
            if self.entries and not confirmed:
                return dict(
                    confirm="queue",
                    message="Start only the currently loaded settings? The queued TOMLs will remain in Run Queue.",
                )
            if self.preferences.values.get("autosave", True) is not False:
                saved = self.save(overwrite=overwrite)
                if saved.get("confirm"):
                    return saved
                path = saved["path"]
            else:
                path = Path(self.storage.name) / f"run-{time.time_ns()}.toml"
                bridge.write_toml(path, flat)
            jobs = (
                toolchain_jobs(path)
                + self.cache_jobs(flat)
                + [Job(train_launch(path, len(self.selected)), training=True)]
            )
        elif action in ("cache", "dry-run"):
            jobs = self.cache_jobs(flat, action == "dry-run")
        elif action == "audit":
            jobs = [
                Job(
                    audit_launch(p, flat.get("dataset.bucket_reso_steps") or 64),
                    on_failure="continue",
                )
                for p in self._paths(flat)
            ]
        elif action == "calibrate":
            self.check(flat)
            path = Path(self.storage.name) / f"calibrate-{time.time_ns()}.toml"
            bridge.write_toml(path, flat)
            jobs = [Job(calibration_launch(path))]
        elif action == "cpp-english":
            if sys.platform != "win32" or not confirmed:
                raise ValueError(
                    "Confirm installation of the Windows English C++ language pack"
                )
            jobs = [
                Job(
                    Launch(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "trainer.tools.windows_toolchain",
                            "--install-english",
                        ],
                        "Install English C++ language pack",
                        is_training=False,
                    )
                )
            ]
        else:
            raise ValueError("Unknown job action")
        if not jobs:
            raise ValueError("Choose an active dataset folder; zero repeats or a zero target disables a folder")
        self._launch_thread(lambda: self._sequence(jobs, flat))
        return self.state()

    def start_queue(self, continue_errors=False):
        if self.busy:
            raise ValueError("A job is already running")
        self.continue_errors = bool(continue_errors)
        reservations = dict(self.reservations)
        for entry in self.entries:
            if entry.status != "Pending":
                continue
            try:
                self._prepare_entry(entry)
            except Exception as exc:
                entry.status, entry.error = "Failed", str(exc)
                if not self.continue_errors:
                    self.reservations = reservations
                    return self.state()
        self.reservations = reservations
        if not any(e.status == "Pending" for e in self.entries):
            return self.state()
        self.queue_active = True
        self._launch_thread(self._queue_loop)
        return self.state()

    def _launch_thread(self, operation):
        self.cancelled.clear()
        self.busy = True

        def run():
            try:
                operation()
            except Exception as exc:
                self.log(f"Job error: {exc}")
            finally:
                with self.lock:
                    if self.queue_active:
                        for entry in self.entries:
                            if entry.status == "Pending":
                                entry.status = "Skipped"
                    self.busy = self.queue_active = False
                    self.worker, self.active = None, None
                self.publish()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def _queue_loop(self):
        while not self.cancelled.is_set():
            with self.lock:
                entry = next((e for e in self.entries if e.status == "Pending"), None)
                if entry is not None:
                    # Reserve the selected entry before potentially slow validation.
                    entry.status = "Caching"
            if entry is None:
                break
            try:
                flat, jobs = self._prepare_entry(entry)
                code = self._sequence(jobs, flat, entry)
                if entry.status != "Completed":
                    entry.status = (
                        "Stopped"
                        if self.cancelled.is_set()
                        else "Failed"
                        if code
                        else "Completed"
                    )
                if entry.status == "Failed":
                    entry.error = self.tail or f"Process exited with code {code}"
                if code and not self.continue_errors:
                    break
            except Exception as exc:
                entry.status, entry.error = "Failed", str(exc)
                if not self.continue_errors:
                    break
            self.publish()

    def _sequence(self, jobs, flat, entry=None):
        for job in jobs:
            if self.cancelled.is_set():
                return -1
            if (
                entry
                and hashlib.sha256(Path(entry.snapshot).read_bytes()).hexdigest()
                != entry.digest
            ):
                raise ValueError(
                    "Queued snapshot was modified; remove it and add the config again."
                )
            with self.lock:
                if entry:
                    entry.status = "Training" if job.training else "Caching"
                self.active = dict(
                    label=job.launch.label,
                    training=job.training,
                    entryId=entry.id if entry else None,
                )
                self.tail = ""
                if job.training:
                    self.metrics = dict(
                        records=[],
                        latest={},
                        reports=[],
                        startedAt=time.time() * 1000,
                        elapsed=None,
                    )
                    self.snapshot_sent = False
                    self.event("metrics_reset", metrics=copy.deepcopy(self.metrics))
                env = training_env(
                    self.gpu_arg(),
                    windows_allocator=self.preferences.values.get(
                        "windows_allocator", "inherit"
                    ),
                )
                env.update(job.launch.env)
                worker = self.worker_factory(job.launch, str(self.root), env)
                self.worker = worker
            result = [-1]
            worker.logSignal.connect(self.log)
            worker.errorSignal.connect(self.log)
            worker.progressSignal.connect(self.log)
            worker.finishedSignal.connect(lambda code: result.__setitem__(0, code))
            worker.metricsSignal.connect(
                lambda line: self._metric_line(line, flat, entry, job.training)
            )
            self.log(f"{job.launch.label}\n" + " ".join(job.launch.argv))
            self.event("activate", panel="metrics" if job.training else "console")
            self.publish()
            # Stop can arrive before Popen creates a PID. ProcessWorker checks the flag.
            if self.cancelled.is_set():
                return -1
            worker.run()
            if job.training:
                with self.lock:
                    started = self.metrics["startedAt"]
                    self.metrics["elapsed"] = (
                        max(0, time.time() * 1000 - started)
                        if started
                        else self.metrics["elapsed"]
                    )
                    self.metrics["startedAt"] = None
                self.event("metrics_finished", elapsed=self.metrics["elapsed"])
                if entry and not result[0] and not self.cancelled.is_set():
                    entry.status = "Completed"
                if (
                    entry
                    and not result[0]
                    and not self.cancelled.is_set()
                    and not flat.get("train.skip_final_save")
                ):
                    self._snapshot(flat)
            self.log(f"Process finished with exit code {result[0]}")
            if self.cancelled.is_set():
                return -1
            if result[0] and job.on_failure == "stop":
                return result[0]
        return 0

    def _metric_line(self, line, flat, entry, training):
        if not training:
            return
        with self.lock:
            if not self.metrics_paused:
                record = parse_step(line)
                if record:
                    previous = self.metrics["latest"].get("ema", record["loss"])
                    record["ema"] = 0.95 * previous + 0.05 * record["loss"]
                    self.metrics["latest"] = record
                    self.metrics["records"].append(record)
                    self.event("metric", record=record)
                    if len(self.metrics["records"]) > 60000:
                        self.metrics["records"] = compact_records(
                            self.metrics["records"]
                        )
                        self.event(
                            "metrics_compact",
                            records=copy.deepcopy(self.metrics["records"]),
                        )
                elif REPORT_RE.match(line):
                    self.metrics["reports"] = (self.metrics["reports"] + [line])[-12:]
                    self.event("report", reports=self.metrics["reports"])
                elif line.startswith("BLUFLOW_STAGE "):
                    try:
                        stage = json.loads(line.removeprefix("BLUFLOW_STAGE "))
                        self.metrics["latest"]["resolution_stage"] = stage
                        self.event("stage", stage=stage)
                    except (ValueError, TypeError):
                        pass
        if (
            entry
            and not self.cancelled.is_set()
            and line.startswith("BLUFLOW_FINAL_SAVE ")
        ):
            try:
                folder = Path(
                    json.loads(line.removeprefix("BLUFLOW_FINAL_SAVE "))["output_dir"]
                ).resolve()
                if folder != self._destination(flat):
                    raise ValueError(
                        "Final-save folder does not match the queued config"
                    )
                self._snapshot(flat)
            except (ValueError, KeyError, TypeError) as exc:
                self.log(f"Could not read final-save notification: {exc}")

    def _snapshot(self, flat):
        if self.snapshot_sent:
            return
        self.snapshot_sent = True
        self.event(
            "metrics_snapshot",
            path=str(self._destination(flat) / "metrics-final.png"),
            metrics=copy.deepcopy(self.metrics),
        )

    def metrics_control(self, action):
        if action == "pause":
            self.metrics_paused = not self.metrics_paused
        elif action == "clear":
            self.metrics.update(records=[], latest={}, reports=[])
            self.event("metrics_reset", metrics=copy.deepcopy(self.metrics))
        else:
            raise ValueError("Unknown metrics action")
        return dict(paused=self.metrics_paused)

    def stop(self):
        self.cancelled.set()
        worker = self.worker
        if worker:
            worker.stop_requested = True
            worker.stop()
        return self.state()

    def close(self):
        self.stop()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=20)
        if not self.thread or not self.thread.is_alive():
            self.storage.cleanup()
