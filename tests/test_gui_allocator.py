"""Allocator choices affect new child processes without changing the GUI environment."""
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from trainer.gui import process


@pytest.mark.skipif(sys.platform != "linux", reason="Linux launch script")
def test_electron_launcher_passes_expandable_segments_to_children(tmp_path):
    node = tmp_path / "node"
    node.write_text('#!/bin/sh\nprintf \'%s\\n\' "${PYTORCH_ALLOC_CONF-unset}" '
                    '"$PYTORCH_CUDA_ALLOC_CONF" "$@"\n')
    node.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "start-electron.sh"), "--no-update-check"],
        cwd=tmp_path,
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
             "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
             "PYTORCH_CUDA_ALLOC_CONF": "backend:native"},
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines() == [
        "unset", "expandable_segments:True", "scripts/electron.cjs", "start",
        "--no-update-check",
    ]


@pytest.mark.parametrize("inherited", [
    {}, {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"},
    {"PYTORCH_CUDA_ALLOC_CONF": "backend:native,max_split_size_mb:128"},
    {"PYTORCH_ALLOC_CONF": ""},
])
def test_windows_default_preserves_user_environment(monkeypatch, inherited):
    monkeypatch.setattr(process, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(process.os, "environ", inherited.copy())
    env = process.training_env("0")
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        assert (key in env) == (key in inherited)
        assert env.get(key) == inherited.get(key)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert process.os.environ == inherited


@pytest.mark.parametrize("selection", ["native", "cudaMallocAsync"])
def test_explicit_choice_overrides_old_launcher_only_in_child(monkeypatch, selection):
    inherited = {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    monkeypatch.setattr(process, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(process.os, "environ", inherited.copy())
    env = process.training_env(windows_allocator=selection)
    assert env["PYTORCH_ALLOC_CONF"] == f"backend:{selection}"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert process.os.environ == inherited
    # Changing preferences affects a future job, not an already constructed env.
    next_env = process.training_env(windows_allocator="native")
    assert next_env["PYTORCH_ALLOC_CONF"] == "backend:native"
    assert env["PYTORCH_ALLOC_CONF"] == f"backend:{selection}"


def test_windows_preference_does_not_affect_linux(monkeypatch):
    inherited = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    monkeypatch.setattr(process, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(process.os, "environ", inherited.copy())
    env = process.training_env(windows_allocator="cudaMallocAsync")
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == inherited["PYTORCH_CUDA_ALLOC_CONF"]
    assert "PYTORCH_ALLOC_CONF" not in env
