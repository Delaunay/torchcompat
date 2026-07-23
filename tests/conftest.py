"""Bootstrap test environment and shared fixtures."""

from __future__ import annotations

import importlib
import os
import sysconfig
from typing import Any

import pytest

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TORCHCOMPAT_SKIP_PLUGINS", "tt,xla,template")


def ensure_libpython_on_path() -> None:
    libdir = sysconfig.get_config_var("LIBDIR")
    if not libdir:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if libdir not in parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([libdir, *parts])


ensure_libpython_on_path()

from torchcompat.utils.device import Device
from torchcompat.utils.errors import NotAvailable

PLUGIN_SPECS: dict[str, dict[str, Any]] = {
    "cpu": {"device_type": "cpu", "name": "cpu", "ccl": "gloo"},
    "cuda": {"device_type": "cuda", "name": "cuda", "ccl": "nccl"},
    "rocm": {"device_type": "cuda", "name": "rocm", "ccl": "nccl"},
    "xpu": {"device_type": "xpu", "name": "xpu", "ccl": "ccl"},
    "gaudi": {"device_type": "hpu", "name": "gaudi", "ccl": "hccl"},
    "xla": {"device_type": "xla", "name": "xla", "ccl": "xla"},
    "tt": {"device_type": "xla", "name": "tt", "ccl": "xla"},
}

# Full Device surface — extras always present via base defaults.
UNIFIED_API = Device.PUBLIC_ATTRS


def plugin_module_name(plugin_name: str) -> str:
    return f"torchcompat.plugins.{plugin_name}"


@pytest.fixture(scope="session")
def plugin_availability():
    availability = {}
    for plugin_name in PLUGIN_SPECS:
        if plugin_name in {"tt", "xla"}:
            availability[plugin_name] = {
                "ok": False,
                "plugin": plugin_name,
                "error": "skipped in unit tests",
                "kind": "Skipped",
            }
            continue
        try:
            module = importlib.import_module(plugin_module_name(plugin_name))
            availability[plugin_name] = {
                "ok": True,
                "plugin": plugin_name,
                "device_type": module.impl.device_type,
                "ccl": module.impl.ccl,
            }
        except NotAvailable as err:
            availability[plugin_name] = {
                "ok": False,
                "plugin": plugin_name,
                "error": str(err),
                "kind": "NotAvailable",
            }
    return availability


@pytest.fixture(params=sorted(PLUGIN_SPECS))
def plugin_name(request):
    return request.param


@pytest.fixture
def plugin_probe(plugin_name, plugin_availability):
    probe = plugin_availability[plugin_name]
    if not probe.get("ok"):
        pytest.skip(probe.get("error", f"{plugin_name} not available"))
    return probe


@pytest.fixture
def plugin_module(plugin_name, plugin_availability):
    if plugin_name in {"tt", "xla"}:
        pytest.skip(f"{plugin_name} is not exercised in unit tests")

    probe = plugin_availability[plugin_name]
    if not probe.get("ok"):
        pytest.skip(probe.get("error", f"{plugin_name} not available"))

    return importlib.import_module(plugin_module_name(plugin_name))


@pytest.fixture
def plugin_impl(plugin_module):
    return plugin_module.impl
