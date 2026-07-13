"""Runtime log configuration for accelerator backends."""

from __future__ import annotations

import os
from pathlib import Path


def log_root() -> Path:
    """Return the torchcompat log directory."""
    override = os.environ.get("TORCHCOMPAT_LOG_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "torchcompat"
    return Path.home() / ".cache" / "torchcompat"


def prepare_tt_environment() -> bool:
    """Configure Tenstorrent env vars before ``torch_xla`` is imported.

    This must run as early as possible on hosts with sysfs-visible TT devices so
    lazy imports configure logging and PJRT before the generic XLA plugin loads.
    """
    from torchcompat.plugins.tt.sysfs import list_sysfs_devices

    if not list_sysfs_devices():
        return False

    os.environ.setdefault("PJRT_DEVICE", "TT")
    os.environ.setdefault("XLA_REGISTER_INSTALLED_PLUGINS", "1")
    os.environ["TT_LOGGER_FILE"] = str(log_root() / "tt" / "logger.txt")
    return True

