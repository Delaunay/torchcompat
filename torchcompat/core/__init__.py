"""Top level module for torchcompat"""

__descr__ = "torch compatibility layer"
__version__ = "1.1.4"
__license__ = "BSD 3-Clause License"
__author__ = "Anonymous"
__author_email__ = "anony@mous.com"
__copyright__ = "2024 Anonymous"
__url__ = "https://github.com/Delaunay/torchcompat"

import sys

from torchcompat.utils.bootstrap import install_eager

install_eager(sys.modules[__name__])
