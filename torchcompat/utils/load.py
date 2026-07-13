"""Plugin loading and backend selection for torchcompat."""

import importlib
import os
import pkgutil
from functools import lru_cache

from .errors import NotAvailable

missing_backend_reason = {}
default_device = None

# Higher priority first. TT is listed before XLA so sysfs-visible Tenstorrent
# hardware is preferred over the generic torch_xla CPU PJRT backend.
BACKEND_PRIORITY = (
    "cuda",
    "rocm",
    "xpu",
    "gaudi",
    "tt",
    "xla",
    "cpu",
)


def _plugin_short_name(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def _select_plugin_impl(plugins: dict, *, allow_cpu: bool = True):
    by_short = {_plugin_short_name(name): module for name, module in plugins.items()}
    for short_name in BACKEND_PRIORITY:
        if short_name == "cpu" and not allow_cpu:
            continue
        module = by_short.get(short_name)
        if module is None:
            continue
        try:
            return module.impl
        except Exception:
            continue
    return None


def _skipped_plugins():
    return {
        name.strip()
        for name in os.environ.get("TORCHCOMPAT_SKIP_PLUGINS", "").split(",")
        if name.strip()
    }


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


def list_plugin_names(*, include_template: bool = False) -> list[str]:
    import torchcompat.plugins

    names = []
    for _, name, _ in pkgutil.iter_modules(torchcompat.plugins.__path__):
        if name == "template" and not include_template:
            continue
        names.append(name)
    return sorted(names)


def discover_plugins(module, *, respect_skip: bool = True):
    """Import available plugins and return ``(plugins, errors)``."""
    from torchcompat.core.logs import prepare_tt_environment

    global missing_backend_reason
    global default_device

    prepare_tt_environment()

    path = module.__path__
    package_name = module.__name__

    plugins = {}
    errors = {}
    skipped = _skipped_plugins() if respect_skip else set()

    discovered = [
        module_name
        for _, module_name, _ in pkgutil.iter_modules(path, package_name + ".")
        if module_name.rsplit(".", 1)[-1] != "template"
    ]
    priority = {name: index for index, name in enumerate(BACKEND_PRIORITY)}
    discovered.sort(
        key=lambda module_name: (
            priority.get(module_name.rsplit(".", 1)[-1], len(BACKEND_PRIORITY)),
            module_name,
        )
    )

    for module_name in discovered:
        short_name = module_name.rsplit(".", 1)[-1]
        if short_name in skipped:
            continue
        try:
            backend = importlib.import_module(module_name)
            if short_name == "cpu":
                default_device = backend

            plugins[module_name] = backend
        except NotAvailable as err:
            errors[module_name] = err
        except Exception as err:
            errors[module_name] = err

    if respect_skip:
        missing_backend_reason.clear()
        missing_backend_reason.update(errors)

    return plugins, errors


def backend_status(*, respect_skip: bool = False) -> dict[str, dict]:
    """Return availability information for every plugin."""
    import torchcompat.plugins

    plugins, errors = discover_plugins(torchcompat.plugins, respect_skip=respect_skip)
    results = {}

    for short_name in list_plugin_names():
        module_name = f"torchcompat.plugins.{short_name}"
        if module_name in plugins:
            impl = plugins[module_name].impl
            result = {
                "ok": True,
                "plugin": short_name,
                "device_type": getattr(impl, "device_type", None),
                "ccl": getattr(impl, "ccl", None),
            }
            if hasattr(impl, "fetch_device"):
                result["device"] = str(impl.fetch_device(0))
            results[short_name] = result
            continue

        err = errors.get(module_name)
        if err is None and respect_skip and short_name in _skipped_plugins():
            results[short_name] = {
                "ok": False,
                "plugin": short_name,
                "error": "skipped",
                "kind": "Skipped",
            }
            continue

        results[short_name] = {
            "ok": False,
            "plugin": short_name,
            "error": str(err) if err else "unavailable",
            "kind": type(err).__name__ if err else "Unknown",
        }

    return results


def load_plugins():
    import torchcompat.plugins

    plugins, _errors = discover_plugins(torchcompat.plugins)
    return plugins


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

    impl = _select_plugin_impl(devices, allow_cpu=False)
    if impl is None:
        explain_errors()

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
    impl = _select_plugin_impl(devices, allow_cpu=True)
    if impl is None:
        explain_errors()

    if ensure is not None:
        assert impl.device_type == ensure

    return impl


if __name__ == "__main__":
    print(load_device())
