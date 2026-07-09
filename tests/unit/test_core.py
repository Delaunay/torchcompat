"""Tests for torchcompat.core unified entry points."""

import os

os.environ.setdefault("TORCHCOMPAT_SKIP_PLUGINS", "tt,xla,template")

import torch
import torch.nn as nn

import torchcompat.core as tc
from torchcompat.core.load import load_available
from torchcompat.core.logs import log_root


def test_core_exports_unified_api():
    for name in (
        "device_module",
        "step",
        "optimizer_step",
        "launch",
        "fetch_device",
        "device_string",
        "mark_step",
        "synchronize",
        "init_process_group",
    ):
        assert hasattr(tc, name)


def test_load_available_cpu():
    os.environ.setdefault("TORCHCOMPAT_SKIP_PLUGINS", "tt,xla,template")
    load_available.cache_clear()
    impl = load_available(ensure="cpu")
    assert impl.device_type == "cpu"


def test_load_available_prefers_tt_over_xla(monkeypatch):
    import types

    tt = types.SimpleNamespace(impl=types.SimpleNamespace(device_type="tt"))
    xla = types.SimpleNamespace(impl=types.SimpleNamespace(device_type="xla"))
    cpu = types.SimpleNamespace(impl=types.SimpleNamespace(device_type="cpu"))

    monkeypatch.setattr(
        "torchcompat.core.load.load_plugins",
        lambda: {
            "torchcompat.plugins.cpu": cpu,
            "torchcompat.plugins.tt": tt,
            "torchcompat.plugins.xla": xla,
        },
    )
    load_available.cache_clear()
    assert load_available().device_type == "tt"


def test_core_step_and_launch():
    with tc.step():
        pass

    seen = {}

    def _fn(rank):
        seen["rank"] = rank

    tc.launch(_fn)
    assert seen["rank"] == 0


def test_mark_step_is_callable():
    tc.mark_step()


def test_log_root_default(monkeypatch, tmp_path):
    monkeypatch.delenv("TORCHCOMPAT_LOG_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert log_root() == tmp_path / ".cache" / "torchcompat"


def test_log_root_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHCOMPAT_LOG_DIR", str(tmp_path / "custom"))
    assert log_root() == tmp_path / "custom"


def test_device_module_matches_load_available():
    os.environ.setdefault("TORCHCOMPAT_SKIP_PLUGINS", "tt,xla,template")
    load_available.cache_clear()
    assert tc.device_module.device_type == load_available(ensure="cpu").device_type
