"""Qt signals/thread lifetime adapter for the shared subprocess implementation.

The worker is derived from Aozora Trainer; see LICENSE.Aozora and NOTICE.
"""

import os
import sys
from PySide6.QtCore import QThread, Signal
from ..ui_core import process as shared
from ..ui_core.process import (
    Launch,
    Job,
    train_launch,
    calibration_launch,
    toolchain_jobs,
    cache_launch,
    concat_launch,
    audit_launch,
)


def training_env(gpus="", *, windows_allocator="inherit"):
    return shared.training_env(
        gpus, windows_allocator=windows_allocator, _platform=sys.platform
    )


class ProcessRunner(QThread):
    logSignal = Signal(str)
    progressSignal = Signal(str, bool)
    finishedSignal = Signal(int)
    errorSignal = Signal(str)
    metricsSignal = Signal(str)
    FATAL_HINTS = shared.ProcessWorker.FATAL_HINTS
    _clean = staticmethod(shared.ProcessWorker._clean)

    def __init__(self, launch, working_dir, env=None):
        super().__init__()
        self.worker = shared.ProcessWorker(launch, working_dir, env)
        self.launch, self.working_dir, self.env = launch, working_dir, self.worker.env
        for name in (
            "logSignal",
            "progressSignal",
            "finishedSignal",
            "errorSignal",
            "metricsSignal",
        ):
            getattr(self.worker, name).connect(getattr(self, name).emit)

    @property
    def process(self):
        return self.worker.process

    @process.setter
    def process(self, value):
        self.worker.process = value

    @property
    def stop_requested(self):
        return self.worker.stop_requested

    @stop_requested.setter
    def stop_requested(self, value):
        self.worker.stop_requested = value

    def run(self):
        self.worker.run()

    def stop(self):
        self.worker.stop()

    def _kill_tree_windows(self):
        return self.worker._kill_tree_windows()
