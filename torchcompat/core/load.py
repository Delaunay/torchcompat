"""Top level module for torchcompat"""

import importlib
import pkgutil
from functools import cache

from torchcompat.core.errors import NotAvailable


def explain_errors(errors):
    frags = []
    for k, v in errors.items():
        message = [str(v)]
        if v.__cause__:
            message.append(f"because")
            message.append(str(v.__cause__))
        error = " ".join(message)

        frags.append(f"{k}: {error}")

    sep = "\n    - "
    errors = sep.join(frags)

    raise NoDeviceDetected(f"Tried:{sep}{errors}")


def discover_plugins(module):
    """Discover uetools plugins"""
    path = module.__path__
    name = module.__name__

    plugins = {}
    errors = {}

    for _, name, _ in pkgutil.iter_modules(path, name + "."):
        try:
            plugins[name] = importlib.import_module(name)
        except NotAvailable as err:
            errors[name] = err

    if len(plugins) == 0:
        explain_errors(errors)

    return plugins


def load_plugins():
    import torchcompat.plugins

    devices = discover_plugins(torchcompat.plugins)

    return devices


class NoDeviceDetected(Exception):
    pass


@cache
def load_device():
    devices = load_plugins()

    return devices.popitem()[1].impl


if __name__ == "__main__":
    # import json
    # import importlib_resources
    # data_path = importlib_resources.files("torchcompat.data")

    # with open(data_path / "data.json", encoding="utf-8") as file:
    #     print(json.dumps(json.load(file), indent=2))

    print(load_device())
