"""Tests for Tenstorrent environment preparation."""

import os
import sys

import pytest


def test_prepare_tt_environment_sets_logger(monkeypatch, tmp_path):
    from torchcompat.core.logs import log_root, prepare_tt_environment

    monkeypatch.setenv("TORCHCOMPAT_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "torchcompat.utils.tt_sysfs.list_sysfs_devices",
        lambda: [{"device_id": 0, "name": "tenstorrent!0"}],
    )

    assert prepare_tt_environment() is True
    assert os.environ["PJRT_DEVICE"] == "TT"
    assert os.environ["TT_LOGGER_FILE"] == str(log_root() / "tt" / "logger.txt")


def test_prepare_tt_environment_no_hardware(monkeypatch):
    from torchcompat.core.logs import prepare_tt_environment

    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    monkeypatch.setattr(
        "torchcompat.utils.tt_sysfs.list_sysfs_devices",
        lambda: [],
    )

    assert prepare_tt_environment() is False
    assert "PJRT_DEVICE" not in os.environ


def test_lazy_import_does_not_prepare_tt_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    monkeypatch.delenv("TT_LOGGER_FILE", raising=False)
    monkeypatch.setenv("TORCHCOMPAT_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "torchcompat.utils.tt_sysfs.list_sysfs_devices",
        lambda: [{"device_id": 0, "name": "tenstorrent!0"}],
    )

    sys.modules.pop("torchcompat.lazy", None)
    import torchcompat.lazy as lazy  # noqa: F401

    assert "PJRT_DEVICE" not in os.environ
    assert "TT_LOGGER_FILE" not in os.environ


def test_lazy_first_access_prepares_tt_environment(monkeypatch, tmp_path):
    from torchcompat.utils.device import Device

    class _Stub(Device):
        @property
        def name(self) -> str:
            return "tt"

        @property
        def device_type(self) -> str:
            return "xla"

        @property
        def ccl(self) -> str:
            return "xla"

    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    monkeypatch.delenv("TT_LOGGER_FILE", raising=False)
    monkeypatch.setenv("TORCHCOMPAT_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "torchcompat.utils.tt_sysfs.list_sysfs_devices",
        lambda: [{"device_id": 0, "name": "tenstorrent!0"}],
    )
    monkeypatch.setattr(
        "torchcompat.utils.load.load_available",
        lambda ensure=None: _Stub(),
    )

    sys.modules.pop("torchcompat.lazy", None)
    import torchcompat.lazy as lazy

    _ = lazy.device_type

    assert os.environ["PJRT_DEVICE"] == "TT"
    assert os.environ["TT_LOGGER_FILE"] == str(tmp_path / "tt" / "logger.txt")


def test_xla_plugin_skips_when_tt_sysfs_present(monkeypatch):
    import importlib

    monkeypatch.setattr(
        "torchcompat.utils.tt_sysfs.list_sysfs_devices",
        lambda: [{"device_id": 0, "name": "tenstorrent!0"}],
    )

    sys.modules.pop("torchcompat.plugins.xla", None)
    with pytest.raises(Exception) as err:
        importlib.import_module("torchcompat.plugins.xla")

    assert "Tenstorrent devices use the TT plugin" in str(err.value)
