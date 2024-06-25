"""Top level module for torchcompat"""

import importlib
import pkgutil
from functools import lru_cache

from torchcompat.core.errors import NotAvailable

missing_backend_reason = {}
default_device = None


class NoDeviceDetected(Exception):
    pass


def explain_errors():
    global missing_backend_reason

    frags = []
    for k, v in missing_backend_reason.items():
        message = [str(v)]
        if v.__cause__:
            message.append("because")
            message.append(str(v.__cause__))
        error = " ".join(message)

        frags.append(f"{k}: {error}")

    sep = "\n    - "
    errors = sep.join(frags)
    raise NoDeviceDetected(f"Tried:{sep}{errors}")


def discover_plugins(module):
    """Discover uetools plugins"""
    global missing_backend_reason
    global default_device

    path = module.__path__
    name = module.__name__

    plugins = {}
    errors = {}

    for _, name, _ in pkgutil.iter_modules(path, name + "."):
        try:
            backend = importlib.import_module(name)
            if "cpu" in name:
                default_device = backend

            plugins[name] = backend
        except NotAvailable as err:
            errors[name] = err

    missing_backend_reason = errors

    return plugins


def load_plugins():
    import torchcompat.plugins

    devices = discover_plugins(torchcompat.plugins)

    return devices


@lru_cache
def load_device(ensure=None):
    """Load a compute device, CPU is not valid.

    Arguments
    ---------
    ensure: optional, str
        name of the expected backend (xpu, cuda, hpu, rocm)
        if the backend do not match raise

    """
    devices = load_plugins()

    if len(devices) == 0:
        explain_errors()

    impl = devices.popitem()[1].impl
    if ensure is not None:
        assert impl.device_type == ensure

    return impl


@lru_cache
def load_available(ensure=None):
    """Load the fastest available compute device, fallsback to CPU

    Arguments
    ---------
    ensure: optional, str
        name of the expected backend (xpu, cuda, hpu, rocm)
        if the backend do not match raise

    """
    devices = load_plugins()

    if len(devices) > 0:
        impl = devices.popitem()[1].impl

    if ensure is not None:
        assert impl.device_type == ensure

    return default_device.impl


if __name__ == "__main__":
    # import json
    # import importlib_resources
    # data_path = importlib_resources.files("torchcompat.data")

    # with open(data_path / "data.json", encoding="utf-8") as file:
    #     print(json.dumps(json.load(file), indent=2))

    print(load_device())
