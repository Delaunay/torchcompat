"""Lazy-loading entry point with the same API as torchcompat.core."""

import sys

from torchcompat.utils.bootstrap import install_lazy

install_lazy(sys.modules[__name__])
