"""Tests for torchcompat.core unified entry points."""

import os

os.environ.setdefault("TORCHCOMPAT_SKIP_PLUGINS", "tt,xla,template")

import torchcompat.core as tc
from torchcompat.core.load import load_available
from torchcompat.core.logs import log_root
from torchcompat.utils.device import Device


def test_core_exports_unified_api():
    for name in (
        "device",
        "device_module",
        "step",
        "optimizer_step",
        "launch",
        "fetch_device",
        "device_string",
        "mark_step",
        "synchronize",
        "init_process_group",
        "init_mesh_group",
        "get_mesh",
        "shard_model",
        "prepare_batch",
    ):
        assert hasattr(tc, name)


def test_core_device_is_device_instance():
    assert isinstance(tc.device, Device)
    assert tc.device is tc.device_module


def test_load_available_cpu():
    os.environ.setdefault("TORCHCOMPAT_SKIP_PLUGINS", "tt,xla,template")
    load_available.cache_clear()
    impl = load_available(ensure="cpu")
    assert isinstance(impl, Device)
    assert impl.device_type == "cpu"
    assert impl.name == "cpu"


def test_load_available_prefers_tt_over_xla(monkeypatch):
    class _Stub(Device):
        def __init__(self, name, device_type):
            self._name = name
            self._device_type = device_type

        @property
        def name(self) -> str:
            return self._name

        @property
        def device_type(self) -> str:
            return self._device_type

        @property
        def ccl(self) -> str:
            return "xla"

    tt = type("M", (), {"impl": _Stub("tt", "xla")})()
    xla = type("M", (), {"impl": _Stub("xla", "xla")})()
    cpu = type("M", (), {"impl": _Stub("cpu", "cpu")})()

    monkeypatch.setattr(
        "torchcompat.utils.load.load_plugins",
        lambda: {
            "torchcompat.plugins.cpu": cpu,
            "torchcompat.plugins.tt": tt,
            "torchcompat.plugins.xla": xla,
        },
    )
    load_available.cache_clear()
    selected = load_available()
    assert selected.name == "tt"
    assert selected.device_type == "xla"
    assert load_available(ensure="tt").name == "tt"
    assert load_available(ensure="xla").device_type == "xla"


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


def test_flat_aliases_match_device():
    assert tc.device_type == tc.device.device_type
    assert tc.ccl == tc.device.ccl
    assert tc.synchronize == tc.device.synchronize


def test_log_root_default(monkeypatch, tmp_path):
    monkeypatch.delenv("TORCHCOMPAT_LOG_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert log_root() == tmp_path / ".cache" / "torchcompat"


def test_log_root_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHCOMPAT_LOG_DIR", str(tmp_path / "custom"))
    assert log_root() == tmp_path / "custom"


def test_device_matches_load_available():
    os.environ.setdefault("TORCHCOMPAT_SKIP_PLUGINS", "tt,xla,template")
    load_available.cache_clear()
    assert tc.device.device_type == load_available(ensure="cpu").device_type
