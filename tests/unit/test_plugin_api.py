"""Unified API tests for each torchcompat plugin."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tests.conftest import PLUGIN_SPECS, UNIFIED_API
from torchcompat.utils.device import Device


def test_plugin_is_device(plugin_impl):
    assert isinstance(plugin_impl, Device)


def test_plugin_metadata(plugin_name, plugin_impl):
    spec = PLUGIN_SPECS[plugin_name]
    assert plugin_impl.device_type == spec["device_type"]
    assert plugin_impl.ccl == spec["ccl"]
    assert plugin_impl.name == spec["name"]


@pytest.mark.parametrize("api_name", UNIFIED_API)
def test_unified_api_present(plugin_impl, api_name):
    assert hasattr(plugin_impl, api_name)
    attr = getattr(plugin_impl, api_name)
    if api_name in ("device_type", "ccl", "name", "Event", "amp", "accelerate"):
        assert attr is not None
    else:
        assert callable(attr)


def test_set_enable_tf32_no_error(plugin_impl):
    plugin_impl.set_enable_tf32(True)
    plugin_impl.set_enable_tf32(False)


def test_step_context_manager(plugin_impl):
    with plugin_impl.step():
        pass


def test_mark_step_no_error(plugin_impl):
    plugin_impl.mark_step()


def test_synchronize_no_error(plugin_impl):
    plugin_impl.synchronize()


def test_get_mesh_initially_none(plugin_impl):
    assert plugin_impl.get_mesh() is None


def test_extras_defaults(plugin_impl):
    assert plugin_impl.is_data_parallel() is False
    assert plugin_impl.is_tensor_parallel() is False
    assert plugin_impl.is_fsdp() is False
    model = nn.Linear(2, 2)
    assert plugin_impl.shard_model(model) is model

    x = torch.zeros(1)
    prepared = plugin_impl.prepare_batch(x)
    assert isinstance(prepared, tuple) and len(prepared) == 1
    assert prepared[0].device.type == plugin_impl.device_type

    named = plugin_impl.prepare_batch(x=x)
    assert set(named) == {"x"}
    assert named["x"].device.type == plugin_impl.device_type


def test_launch_invokes_fn(plugin_impl):
    seen = {}

    def _fn(rank, value):
        seen["rank"] = rank
        seen["value"] = value

    plugin_impl.launch(_fn, args=(42,))
    assert seen["rank"] == 0
    assert seen["value"] == 42


def test_fetch_device(plugin_impl):
    device = plugin_impl.fetch_device(0)
    assert isinstance(device, torch.device)


def test_device_string(plugin_name, plugin_impl):
    expected_prefix = PLUGIN_SPECS[plugin_name]["device_type"]
    assert plugin_impl.device_string(0).startswith(f"{expected_prefix}:")


def test_compile_returns_callable_module(plugin_name, plugin_impl):
    model = nn.Linear(4, 2)
    compiled = plugin_impl.compile(model)

    if plugin_name in ("xla", "tt"):
        assert compiled is not None
    else:
        assert compiled is not None


def test_optimizer_step_runs(plugin_name, plugin_impl):
    device = plugin_impl.fetch_device(0)

    model = nn.Linear(4, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 4, device=device)
    loss = model(x).sum()

    if plugin_name in ("xla", "tt"):
        with plugin_impl.step():
            loss.backward()
            plugin_impl.optimizer_step(optimizer)
        plugin_impl.synchronize()
    else:
        loss.backward()
        plugin_impl.optimizer_step(optimizer)


def test_init_mesh_group_mesh_requires_axis_names(plugin_impl):
    if plugin_impl.name != "tt":
        # No-op on non-TT backends.
        assert plugin_impl.init_mesh_group(mesh_shape=(1, 1)) is None
        return
    with pytest.raises(ValueError, match="mesh_axis_names"):
        plugin_impl.init_mesh_group(mesh_shape=(1, 1))


@pytest.mark.timeout(60)
def test_linear_forward_on_device(plugin_name, plugin_impl):
    device = plugin_impl.fetch_device(0)

    model = nn.Linear(4, 2).to(device)
    x = torch.randn(2, 4, device=device)

    if plugin_name in ("xla", "tt"):
        with plugin_impl.step():
            y = model(x)
        plugin_impl.mark_step()
    else:
        y = model(x)

    assert y.shape == (2, 2)
