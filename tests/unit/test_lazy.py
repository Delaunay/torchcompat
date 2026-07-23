"""Tests for torchcompat.lazy deferred backend loading."""

import os
import sys

import pytest

os.environ.setdefault("TORCHCOMPAT_SKIP_PLUGINS", "tt,xla,template")


def _reload_lazy():
    sys.modules.pop("torchcompat.lazy", None)
    import torchcompat.lazy as lazy

    return lazy


def test_lazy_import_does_not_load_backend(monkeypatch):
    called = {"count": 0}

    def _fake_load_available(*args, **kwargs):
        called["count"] += 1
        from torchcompat.core.load import load_available

        load_available.cache_clear()
        return load_available(*args, **kwargs)

    monkeypatch.setattr(
        "torchcompat.utils.load.load_available",
        _fake_load_available,
    )

    lazy = _reload_lazy()
    assert called["count"] == 0
    assert lazy._tc_device_loaded is False


def test_lazy_loads_backend_on_first_attribute_access(monkeypatch):
    called = {"count": 0}
    real_load = None

    def _counting_load_available(*args, **kwargs):
        nonlocal real_load
        if real_load is None:
            from torchcompat.core.load import load_available as _load

            real_load = _load
        called["count"] += 1
        real_load.cache_clear()
        return real_load(*args, **kwargs)

    monkeypatch.setattr(
        "torchcompat.utils.load.load_available",
        _counting_load_available,
    )

    lazy = _reload_lazy()
    assert lazy.step is not None
    assert called["count"] == 1
    assert lazy._tc_device_loaded is True
    assert lazy.device.device_type == "cpu"


def test_lazy_exports_same_api_as_core():
    import torchcompat.core as core

    lazy = _reload_lazy()

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
        "Event",
        "get_mesh",
        "prepare_batch",
    ):
        assert hasattr(lazy, name), name
        assert callable(getattr(lazy, name)) or name in (
            "device",
            "device_module",
            "Event",
        )

    with lazy.step():
        pass

    assert lazy.device.device_type == core.device.device_type


def test_lazy_device_string_triggers_load(monkeypatch):
    called = {"count": 0}
    real_load = None

    def _counting_load_available(*args, **kwargs):
        nonlocal real_load
        if real_load is None:
            from torchcompat.core.load import load_available as _load

            real_load = _load
        called["count"] += 1
        real_load.cache_clear()
        return real_load(*args, **kwargs)

    monkeypatch.setattr(
        "torchcompat.utils.load.load_available",
        _counting_load_available,
    )

    lazy = _reload_lazy()
    assert lazy.device_string(0) == "cpu:0"
    assert called["count"] == 1
