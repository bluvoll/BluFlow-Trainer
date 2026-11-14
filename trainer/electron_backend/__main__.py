"""Line-delimited local RPC. stdout is reserved exclusively for protocol messages."""

from __future__ import annotations
import argparse
import contextlib
import json
import math
import sys
import threading
import uuid
from dataclasses import asdict
from pathlib import Path


def wire_value(value):
    """Keep config collections structured and invalid floats representable in JSON."""
    if isinstance(value, dict):
        return {key: wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire_value(item) for item in value]
    if isinstance(value, set):
        return [wire_value(item) for item in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class Server:
    def __init__(self, session, write):
        self.session, self.write = session, write
        self.utilities = {}
        self.methods = {
            "bootstrap": lambda p: session.bootstrap(),
            "state": lambda p: session.state(),
            "config.open": lambda p: session.open(p["path"]),
            "config.edit": lambda p: session.edit(p["patch"]),
            "config.save": lambda p: session.save(
                p.get("destination"), p.get("overwrite", False)
            ),
            "preferences.set": lambda p: session.set_preferences(p["patch"]),
            "gpus.set": lambda p: session.select_gpus(p["selected"]),
            "preview.timestep": lambda p: session.set_preview(p["height"], p["width"]),
            "queue.add": lambda p: session.add_queue(p["paths"]),
            "queue.edit": lambda p: session.queue_edit(
                p["operation"], p.get("ids", []), p.get("direction", 0)
            ),
            "queue.start": lambda p: session.start_queue(
                p.get("continueErrors", False)
            ),
            "job.start": lambda p: session.start(
                p.get("action", "train"),
                p.get("confirmed", False),
                p.get("overwrite", False),
            ),
            "job.stop": lambda p: session.stop(),
            "metrics.get": lambda p: session.metrics,
            "console.read": lambda p: session.console_read(
                p.get("start"), p.get("count", 900)
            ),
            "console.text": lambda p: session.console_text(),
            "console.clear": lambda p: session.console_clear(),
            "metrics.control": lambda p: session.metrics_control(p["action"]),
            "metrics.exportResult": self.export_result,
            "utility.run": self.utility,
            "startup.check": self.startup_check,
            "startup.update": self.startup_update,
        }

    def export_result(self, params):
        if params.get("error"):
            self.session.log(
                "Could not save final metrics screenshot: " + str(params["error"])
            )
            self.session.snapshot_sent = False
        else:
            self.session.log(
                "Saved final metrics screenshot: " + str(params.get("path", ""))
            )
        return True

    def startup_check(self, params):
        from ..gui.startup import check_update
        from ..tools.update_dependencies import dependencies_current, in_venv

        result = {}
        if not params.get("skipUpdate"):
            try:
                result["update"] = asdict(check_update(self.session.root))
            except Exception as exc:
                self.session.log(f"Update check skipped: {exc}")
        if not params.get("skipDependencies"):
            result["refreshDependencies"] = in_venv() and not dependencies_current(
                self.session.root
            )
        return result

    def startup_update(self, params):
        if self.session.busy or self.utilities:
            raise ValueError("Finish running jobs and utilities before updating")
        from ..gui.startup import Update, apply_update

        if params.get("update"):
            apply_update(self.session.root, Update(**params["update"]))
        return True

    def utility(self, params):
        from ..ui_core.process import Launch, ProcessWorker, training_env

        if params.get("kind") not in (
            "alpha-inspect",
            "alpha",
            "qkv",
            "resize",
            "recover",
        ):
            raise ValueError("Unknown utility")
        if params["kind"] == "alpha":
            args = params.get("args", {})
            target = Path(args["destination"]).resolve()
            if target == Path(args["source"]).resolve():
                raise ValueError(
                    "The source file is protected. Choose a different output filename."
                )
            if target.exists() and not args.get("overwrite"):
                return dict(
                    confirm="overwrite",
                    message=f"{target.name} already exists. Overwrite this output copy?",
                )
        token = uuid.uuid4().hex
        path = Path(self.session.storage.name) / f"utility-{token}.json"
        path.write_text(json.dumps(params, ensure_ascii=True), encoding="utf-8")
        worker = ProcessWorker(
            Launch(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "trainer.electron_backend.utility",
                    str(path),
                ],
                params["kind"],
                is_training=False,
            ),
            str(self.session.root),
            training_env(),
        )
        complete = [False]

        def output(text):
            if text.startswith("BLUFLOW_UTILITY_RESULT "):
                result = json.loads(text.removeprefix("BLUFLOW_UTILITY_RESULT "))
                complete[0] = True
                self.session.event("utility_result", id=token, **result)
            else:
                self.session.event("utility_log", id=token, text=text)

        worker.metricsSignal.connect(output)
        worker.logSignal.connect(
            lambda text: self.session.event("utility_log", id=token, text=text)
        )
        worker.errorSignal.connect(
            lambda text: self.session.event("utility_log", id=token, text=text)
        )

        def finished(code):
            if not complete[0]:
                self.session.event(
                    "utility_result",
                    id=token,
                    error=f"Utility process exited with code {code}",
                )
            self.utilities.pop(token, None)
            path.unlink(missing_ok=True)

        worker.finishedSignal.connect(finished)
        self.utilities[token] = worker
        worker.start()
        return dict(id=token)

    def close(self):
        for worker in list(self.utilities.values()):
            worker.stop_requested = True
            worker.stop()
            worker.join(timeout=10)
        self.session.close()

    def handle(self, request):
        method = request.get("method")
        if method not in self.methods:
            raise ValueError("Unknown desktop operation")
        params = request.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("Operation parameters must be an object")
        return self.methods[method](params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir")
    parser.add_argument(
        "--preferences", help="Override preference path for isolated testing"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = sys.stdout
    mutex = threading.Lock()

    def write(message):
        with mutex:
            output.write(
                json.dumps(wire_value(message), ensure_ascii=True, allow_nan=False)
                + "\n"
            )
            output.flush()

    # Third-party import messages and warnings must never corrupt the stdio protocol.
    with contextlib.redirect_stdout(sys.stderr):
        from ..ui_core.session import Session
        from ..ui_core.preferences import Preferences

        if args.check:
            from ..ui_core.config import descriptors
            from ..gui.bridge import missing_ui_keys

            definitions = descriptors()
            missing, unknown = missing_ui_keys(definitions["fields"])
            if missing or unknown or any(k.startswith("PySide6") for k in sys.modules):
                raise RuntimeError(
                    f"Headless coverage check failed: {missing=}, {unknown=}"
                )
            write(dict(ok=True, fields=len(definitions["fields"]), qtImported=False))
            return
        session = Session(
            write, preferences=Preferences(args.preferences), config_dir=args.config_dir
        )
        server = Server(session, write)
        try:
            for line in sys.stdin:
                request = {}
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError("Request must be an object")
                    if request.get("method") == "shutdown":
                        break
                    write(dict(id=request.get("id"), result=server.handle(request)))
                except Exception as exc:
                    write(
                        dict(
                            id=request.get("id") if isinstance(request, dict) else None,
                            error=dict(message=f"{type(exc).__name__}: {exc}"),
                        )
                    )
        finally:
            server.close()


if __name__ == "__main__":
    main()
