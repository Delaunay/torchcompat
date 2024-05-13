"""Plugin example"""

import torch

from torchcompat.core.errors import NotAvailable

impl = torch.cpu


def set_enable_tf32(enable=True):
    pass


ccl = "gloo"
setattr(impl, "device_type", "cpu")
setattr(impl, "set_enable_tf32", set_enable_tf32)
setattr(impl, "ccl", ccl)
