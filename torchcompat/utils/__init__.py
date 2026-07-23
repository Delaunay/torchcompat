"""torchcompat utilities."""

from .bootstrap import install_eager, install_lazy
from .device import Device
from .load import load_available

__all__ = ["Device", "install_eager", "install_lazy", "load_available"]
