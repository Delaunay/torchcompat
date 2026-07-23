"""Shared bootstrap for torchcompat.core and torchcompat.lazy."""

from __future__ import annotations

import types
from typing import Any

from torchcompat.utils.device import Device, local_rank


def install_eager(module: types.ModuleType) -> None:
    """Install the unified API and load the active backend immediately."""
    from torchcompat.utils.load import load_available

    _install_module_helpers(module)
    _bind_device(module, load_available())
    module._tc_device_loaded = True


def install_lazy(module: types.ModuleType) -> None:
    """Install the unified API and defer backend loading until first use."""
    _install_module_helpers(module)
    module._tc_device_loaded = False

    def __getattr__(name: str) -> Any:
        device = _ensure_device_loaded(module)
        if name in vars(module):
            return vars(module)[name]
        try:
            return getattr(device, name)
        except AttributeError as err:
            raise AttributeError(
                f"module {module.__name__!r} has no attribute {name!r}"
            ) from err

    def __dir__() -> list[str]:
        names = {name for name in vars(module) if not name.startswith("_")}
        names.update(Device.PUBLIC_ATTRS)
        names.add("device")
        names.add("device_module")
        if module._tc_device_loaded:
            names.update(
                name
                for name in dir(module.device)
                if not name.startswith("_")
            )
        return sorted(names)

    module.__getattr__ = __getattr__  # noqa: A001
    module.__dir__ = __dir__


def _ensure_device_loaded(module: types.ModuleType) -> Device:
    if module._tc_device_loaded:
        return module.device

    from torchcompat.core.logs import prepare_tt_environment
    from torchcompat.utils.load import load_available

    prepare_tt_environment()
    device = load_available()
    _bind_device(module, device)
    module._tc_device_loaded = True
    return device


def _get_device(module: types.ModuleType) -> Device:
    if module._tc_device_loaded:
        return module.device
    return _ensure_device_loaded(module)


def _bind_device(module: types.ModuleType, device: Device) -> None:
    module.device = device
    # Backward-compatible alias used by older tests / callers.
    module.device_module = device

    for name in Device.PUBLIC_ATTRS:
        setattr(module, name, getattr(device, name))

    # Forward remaining backend extras (e.g. torch.cuda.max_memory_allocated).
    def __getattr__(name: str) -> Any:
        try:
            return getattr(module.device, name)
        except AttributeError as err:
            raise AttributeError(
                f"module {module.__name__!r} has no attribute {name!r}"
            ) from err

    module.__getattr__ = __getattr__  # noqa: A001


def _install_module_helpers(module: types.ModuleType) -> None:
    """Helpers that live on the module, not on Device."""

    def fetch_device_id() -> int:
        return local_rank()

    def optimize(model, *args, optimizer=None, dtype=None, **kwargs):
        if dtype is not None:
            pass

        if optimizer is None:
            return model

        return model, optimizer

    module.fetch_device_id = fetch_device_id
    module.optimize = optimize
