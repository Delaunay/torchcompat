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
