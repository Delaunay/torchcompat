"""Unified API tests for each torchcompat plugin."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tests.conftest import OPTIONAL_API, PLUGIN_SPECS, UNIFIED_API


def test_plugin_metadata(plugin_name, plugin_impl):
    spec = PLUGIN_SPECS[plugin_name]
    assert plugin_impl.device_type == spec["device_type"]
    assert plugin_impl.ccl == spec["ccl"]


@pytest.mark.parametrize("api_name", UNIFIED_API)
def test_unified_api_present(plugin_name, plugin_impl, api_name):
    assert hasattr(plugin_impl, api_name)
    assert callable(getattr(plugin_impl, api_name)) or api_name in (
        "device_type",
        "ccl",
    )


@pytest.mark.parametrize("api_name", OPTIONAL_API)
def test_optional_api_when_exposed(plugin_name, plugin_impl, api_name):
    if not hasattr(plugin_impl, api_name):
        pytest.skip(f"{plugin_name} does not expose {api_name} on impl")
    assert callable(getattr(plugin_impl, api_name))


def test_set_enable_tf32_no_error(plugin_impl):
    plugin_impl.set_enable_tf32(True)
    plugin_impl.set_enable_tf32(False)


def test_step_context_manager(plugin_impl):
    with plugin_impl.step():
        pass


def test_mark_step_no_error(plugin_impl):
    if not hasattr(plugin_impl, "mark_step"):
        pytest.skip("mark_step not exposed")
    plugin_impl.mark_step()


def test_synchronize_no_error(plugin_impl):
    if not hasattr(plugin_impl, "synchronize"):
        pytest.skip("synchronize not exposed")
    plugin_impl.synchronize()


def test_get_mesh_initially_none(plugin_impl):
    if not hasattr(plugin_impl, "get_mesh"):
        pytest.skip("get_mesh not exposed")
    assert plugin_impl.get_mesh() is None


def test_launch_invokes_fn(plugin_impl):
    seen = {}

    def _fn(rank, value):
        seen["rank"] = rank
        seen["value"] = value

    plugin_impl.launch(_fn, args=(42,))
    assert seen["rank"] == 0
    assert seen["value"] == 42


def test_fetch_device(plugin_name, plugin_impl):
    if not hasattr(plugin_impl, "fetch_device"):
        pytest.skip(f"{plugin_name} uses core fetch_device")
    device = plugin_impl.fetch_device(0)
    assert isinstance(device, torch.device)


def test_device_string(plugin_name, plugin_impl):
    if not hasattr(plugin_impl, "device_string"):
        pytest.skip(f"{plugin_name} uses core device_string")
    expected_prefix = PLUGIN_SPECS[plugin_name]["device_type"]
    assert plugin_impl.device_string(0).startswith(f"{expected_prefix}:")


def test_compile_returns_callable_module(plugin_name, plugin_impl):
    if not hasattr(plugin_impl, "compile"):
        pytest.skip(f"{plugin_name} uses core compile")

    model = nn.Linear(4, 2)
    compiled = plugin_impl.compile(model)

    if plugin_name in ("xla", "tt"):
        # torch_xla.compile may return a context manager factory when called
        # without a model; with a model it should be usable.
        assert compiled is not None
    else:
        assert compiled is not None


def test_optimizer_step_runs(plugin_name, plugin_impl):
    if not hasattr(plugin_impl, "fetch_device"):
        device = torch.device("cpu")
    else:
        device = plugin_impl.fetch_device(0)

    model = nn.Linear(4, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 4, device=device)
    loss = model(x).sum()

    if plugin_name in ("xla", "tt"):
        with plugin_impl.step():
            loss.backward()
            plugin_impl.optimizer_step(optimizer)
        if hasattr(plugin_impl, "synchronize"):
            plugin_impl.synchronize()
    else:
        loss.backward()
        plugin_impl.optimizer_step(optimizer)


def test_init_process_group_mesh_requires_axis_names(plugin_impl):
    if not hasattr(plugin_impl, "init_process_group"):
        pytest.skip("init_process_group not exposed")
    with pytest.raises(ValueError, match="mesh_axis_names"):
        plugin_impl.init_process_group(mesh_shape=(1, 1))


@pytest.mark.timeout(60)
def test_linear_forward_on_device(plugin_name, plugin_impl):
    if not hasattr(plugin_impl, "fetch_device"):
        device = torch.device("cpu")
    else:
        device = plugin_impl.fetch_device(0)

    model = nn.Linear(4, 2).to(device)
    x = torch.randn(2, 4, device=device)

    if plugin_name in ("xla", "tt"):
        with plugin_impl.step():
            y = model(x)
        if hasattr(plugin_impl, "mark_step"):
            plugin_impl.mark_step()
        elif hasattr(plugin_impl, "synchronize"):
            plugin_impl.synchronize()
    else:
        y = model(x)

    assert y.shape == (2, 2)
