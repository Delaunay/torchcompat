"""Shared bootstrap for torchcompat.core and torchcompat.lazy."""

from __future__ import annotations

import types
from typing import Any


def install_eager(module: types.ModuleType) -> None:
    """Install the unified API and load the active backend immediately."""
    from torchcompat.utils.load import load_available

    _install_base_api(module, lazy=False)
    _bind_device_module(module, load_available())
    module._tc_device_loaded = True


def install_lazy(module: types.ModuleType) -> None:
    """Install the unified API and defer backend loading until first use."""
    _install_base_api(module, lazy=True)
    module._tc_device_loaded = False

    def __getattr__(name: str) -> Any:
        _ensure_device_loaded(module)
        if name in vars(module):
            return vars(module)[name]
        raise AttributeError(
            f"module {module.__name__!r} has no attribute {name!r}"
        )

    def __dir__() -> list[str]:
        names = {name for name in vars(module) if not name.startswith("_")}
        if module._tc_device_loaded:
            names.update(
                key for key in vars(module.device_module) if not key.startswith("_")
            )
        else:
            names.add("device_module")
        return sorted(names)

    module.__getattr__ = __getattr__  # noqa: A001
    module.__dir__ = __dir__


def _ensure_device_loaded(module: types.ModuleType) -> Any:
    if module._tc_device_loaded:
        return module.device_module

    from torchcompat.core.logs import prepare_tt_environment
    from torchcompat.utils.load import load_available

    prepare_tt_environment()
    device_module = load_available()
    _bind_device_module(module, device_module)
    _install_plugin_fallbacks(module)
    module._tc_device_loaded = True
    return device_module


def _get_device_module(module: types.ModuleType) -> Any:
    if module._tc_device_loaded:
        return module.device_module
    return _ensure_device_loaded(module)


def _bind_device_module(module: types.ModuleType, device_module: Any) -> None:
    module.device_module = device_module
    for key, value in vars(device_module).items():
        if key.startswith("__"):
            continue
        setattr(module, key, value)


def _install_base_api(module: types.ModuleType, *, lazy: bool) -> None:
    def fetch_device_id() -> int:
        try:
            import os

            return int(os.getenv("LOCAL_RANK", "0"))
        except Exception:
            return 0

    def device_string(id: int = fetch_device_id()) -> str:
        return f"{_get_device_module(module).device_type}:{id}"

    def mark_step() -> None:
        _noop()

    def fetch_device(id: int = fetch_device_id()):
        import torch

        return torch.device(device_string(id))

    def init_process_group(
        *args, backend=None, rank=-1, world_size=-1, **kwargs
    ) -> None:
        import torch

        backend = backend or _get_device_module(module).ccl
        torch.distributed.init_process_group(
            *args, backend=backend, rank=rank, world_size=world_size, **kwargs
        )

    def destroy_process_group() -> None:
        import torch

        torch.distributed.destroy_process_group()

    def set_enable_tf32(enable=True) -> None:
        _noop_enable_tf32(enable)

    def optimize(model, *args, optimizer=None, dtype=None, **kwargs):
        if dtype is not None:
            pass

        if optimizer is None:
            return model
        return model, optimizer

    def empty_cache() -> None:
        pass

    def synchronize() -> None:
        _noop()

    def is_available() -> bool:
        return True

    class accelerate:
        def Accelerator(*args, **kwargs):
            from accelerate import Accelerator

            return Accelerator(*args, **kwargs)

    module.fetch_device_id = fetch_device_id
    module.device_string = device_string
    module.fetch_device = fetch_device
    module.init_process_group = init_process_group
    module.destroy_process_group = destroy_process_group
    module.optimize = optimize
    module.empty_cache = empty_cache
    module.is_available = is_available
    module.accelerate = accelerate

    if lazy:
        return

    module.mark_step = mark_step
    module.set_enable_tf32 = set_enable_tf32
    module.synchronize = synchronize


def _install_plugin_fallbacks(module: types.ModuleType) -> None:
    names = vars(module)
    if "mark_step" not in names:
        module.mark_step = _noop
    if "set_enable_tf32" not in names:
        module.set_enable_tf32 = _noop_enable_tf32
    if "synchronize" not in names:
        module.synchronize = _noop


def _noop() -> None:
    pass


def _noop_enable_tf32(enable=True) -> None:
    pass
