"""Plugin example"""

import torch

from torchcompat.core.errors import NotAvailable

try:
    import habana_frameworks.torch.core as htcore
except ImportError as err:
    raise NotAvailable("Could not import habana_framworks") from err


impl = htcore
