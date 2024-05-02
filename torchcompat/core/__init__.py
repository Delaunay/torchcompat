"""Top level module for torchcompat"""


__descr__ = "torch compatibility layer"
__version__ = "0.0.1"
__license__ = "BSD 3-Clause License"
__author__ = "Anonymous"
__author_email__ = "anony@mous.com"
__copyright__ = "2024 Anonymous"
__url__ = "https://github.com/Delaunay/torchcompat"


import sys

from torchcompat.core.load import load_device

device_module = load_device()


self = current_module = sys.modules[__name__]
for k, v in vars(device_module).items():
    setattr(self, k, v)
