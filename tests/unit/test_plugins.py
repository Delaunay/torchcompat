"""Plugin discovery tests."""

import importlib

import pytest

import torchcompat.plugins
from torchcompat.utils.errors import NotAvailable
from torchcompat.core.load import discover_plugins, missing_backend_reason

from tests.conftest import PLUGIN_SPECS, plugin_module_name


def _discover():
    return discover_plugins(torchcompat.plugins)


def test_cpu_plugin_always_available(plugin_availability):
    assert plugin_availability["cpu"]["ok"]


def test_plugin_availability_report(plugin_availability):
    available = [name for name, probe in plugin_availability.items() if probe.get("ok")]
    assert "cpu" in available
    assert len(available) >= 1


def test_unavailable_plugins_raise_not_available():
    _, errors = _discover()
    for plugin_name in PLUGIN_SPECS:
        if plugin_name == "cpu":
            continue
        key = plugin_module_name(plugin_name)
        if key in errors:
            assert isinstance(errors[key], NotAvailable)


def test_try_import_in_process():
    cpu = importlib.import_module(plugin_module_name("cpu"))
    assert cpu.impl.device_type == "cpu"
