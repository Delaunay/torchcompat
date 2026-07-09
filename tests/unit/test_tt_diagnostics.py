from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SYSFS_PATH = (
    Path(__file__).resolve().parents[2]
    / "torchcompat"
    / "plugins"
    / "tt"
    / "sysfs.py"
)


def _load_sysfs_module():
    spec = spec_from_file_location("torchcompat_tt_sysfs_test", SYSFS_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_format_hugepage_error():
    sysfs = _load_sysfs_module()
    err = RuntimeError(
        "Failed to pin pages for hugepage at virtual address 0x7fe840000000: "
        "Cannot allocate memory"
    )
    message = sysfs.format_init_error(err)
    assert "hugepages" in message.lower()
    assert "fresh Python" in message


def test_format_reinit_error():
    sysfs = _load_sysfs_module()
    err = RuntimeError("InitializeComputationClient() can only be called once.")
    message = sysfs.format_init_error(err)
    assert "Restart Python" in message


def test_list_sysfs_devices_live():
    sysfs = _load_sysfs_module()
    devices = sysfs.list_sysfs_devices()
    if not Path("/sys/class/tenstorrent").is_dir():
        pytest.skip("no Tenstorrent sysfs on this host")
    assert len(devices) >= 1
    assert devices[0]["name"].startswith("tenstorrent!")


def test_format_sysfs_summary_empty():
    sysfs = _load_sysfs_module()
    assert "no Tenstorrent devices" in sysfs.format_sysfs_summary([])


def test_validate_visible_devices_missing():
    sysfs = _load_sysfs_module()
    devices = [{"device_id": 0, "name": "tenstorrent!0"}]
    message = sysfs.validate_visible_devices("0,9", devices)
    assert message is not None
    assert "9" in message


def test_hardware_unavailable_message_no_class_dir(tmp_path, monkeypatch):
    sysfs = _load_sysfs_module()
    missing = tmp_path / "missing"
    monkeypatch.setattr(sysfs, "SYSFS_TT_CLASS", missing)
    message = sysfs.hardware_unavailable_message()
    assert "/sys/class/tenstorrent" in message
