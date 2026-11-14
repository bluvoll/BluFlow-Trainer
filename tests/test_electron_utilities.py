"""Electron utility operations use the original tensor/file implementations."""

import hashlib
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file, load_file
from trainer.electron_backend.utility import operate
from trainer.modeling.checkpoint_metadata import encode_settings
from trainer.gui import bridge
from trainer.training.config import load_config


@pytest.fixture
def adapter(tmp_path):
    source = tmp_path / "adapter.safetensors"
    save_file(
        {
            "layer.lora_down.weight": torch.arange(8, dtype=torch.float32).reshape(
                2, 4
            ),
            "layer.lora_up.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "layer.alpha": torch.tensor(1.0),
        },
        source,
        metadata={"test": "preserved"},
    )
    return source


def test_alpha_inspect_bake_and_source_protection(adapter, tmp_path):
    before = hashlib.sha256(adapter.read_bytes()).hexdigest()
    info = operate("alpha-inspect", dict(source=str(adapter)))
    assert info["groups"] == [(2, 1.0, 1)]
    target = tmp_path / "baked.safetensors"
    operate(
        "alpha",
        dict(source=str(adapter), destination=str(target), mode="bake", value=1.5),
    )
    output = load_file(target)
    assert output["layer.alpha"].item() == 1.5
    assert torch.equal(
        output["layer.lora_up.weight"], load_file(adapter)["layer.lora_up.weight"]
    )
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError):
        operate("alpha", dict(source=str(adapter), destination=str(adapter)))
    with pytest.raises(ValueError):
        operate("alpha", dict(source=str(adapter), destination=str(target)))


def test_resize_defaults_and_new_destination(adapter, tmp_path):
    result = operate(
        "resize",
        dict(
            source=str(adapter),
            destination=str(tmp_path / "small.safetensors"),
            method="sv_fro",
            threshold=0.99,
            rank=None,
        ),
    )
    assert Path(result["path"]).is_file()
    assert load_file(result["path"])["layer.lora_down.weight"].shape[0] <= 2
    with pytest.raises(ValueError):
        operate("resize", dict(source=str(adapter), destination=result["path"]))


def test_qkv_roundtrip_preserves_dense_updates(tmp_path):
    tensors = {}
    for index, projection in enumerate(("q", "k", "v")):
        prefix = "diffusion_model.layers.0.proj." + projection
        tensors[prefix + ".lora_down.weight"] = (
            torch.arange(8, dtype=torch.float32).reshape(2, 4) + index
        )
        tensors[prefix + ".lora_up.weight"] = torch.arange(
            8, dtype=torch.float32
        ).reshape(4, 2)
        tensors[prefix + ".alpha"] = torch.tensor(1.0)
    source = tmp_path / "qkv.safetensors"
    save_file(tensors, source)
    fused = operate("qkv", dict(source=str(source), layout="fused"))
    separated = operate("qkv", dict(source=fused["path"], layout="separated"))
    restored = load_file(separated["path"])
    for p in ("q", "k", "v"):
        prefix = "diffusion_model.layers.0.proj." + p

        def delta(state):
            down = state[prefix + ".lora_down.weight"]
            return (
                state[prefix + ".lora_up.weight"]
                @ down
                * (state[prefix + ".alpha"] / down.shape[0])
            )

        torch.testing.assert_close(delta(restored), delta(tensors))


def test_recover_metadata_without_resume(tmp_path):
    config = tmp_path / "original.toml"
    bridge.write_toml(
        config,
        bridge.defaults()
        | {
            "dataset.path": "画像/dataset",
            "train.model_family": "moss_image",
            "adapter.kind": "lora",
        },
    )
    source = tmp_path / "saved.safetensors"
    save_file(
        {"placeholder": torch.ones(1)},
        source,
        metadata={"training_config": encode_settings(load_config(config))},
    )
    result = operate("recover", dict(source=str(source)))
    recovered = load_config(result["path"])
    assert recovered.dataset.path == "画像/dataset"
    assert not recovered.train.resume_from


def test_rpc_utility_streams_result_while_command_pipe_stays_open(adapter, tmp_path):
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "trainer.electron_backend",
            "--preferences",
            str(tmp_path / "gui.json"),
            "--config-dir",
            str(tmp_path / "configs"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    messages = queue.Queue()

    def read():
        for line in process.stdout:
            messages.put(json.loads(line))

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        process.stdin.write(
            json.dumps(
                dict(
                    id=1,
                    method="utility.run",
                    params=dict(kind="alpha-inspect", args=dict(source=str(adapter))),
                )
            )
            + "\n"
        )
        process.stdin.flush()
        result = None
        for _ in range(10):
            message = messages.get(timeout=20)
            if message.get("event") == "utility_result":
                result = message
                break
        assert result is not None
        assert "error" not in result, result
        assert result["result"]["modules"] == 1
        assert process.poll() is None
    finally:
        process.stdin.write('{"method":"shutdown"}\n')
        process.stdin.flush()
        process.wait(timeout=20)
        reader.join(timeout=5)
