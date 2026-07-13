"""torchcompat utilities."""

from .bootstrap import install_eager, install_lazy
from .load import load_available

__all__ = ["install_eager", "install_lazy", "load_available"]
