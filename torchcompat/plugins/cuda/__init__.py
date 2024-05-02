"""CUDA compatibility layer"""

import torch

from torchcompat.core.errors import NotAvailable

if not torch.cuda.is_available():
    raise NotAvailable("torch.cuda is not available")

# check that torch.cuda is infact cuda and NOT rocm
if not torch.version.cuda:
    raise NotAvailable("torch.cuda is not rocm")


impl = torch.cuda

setattr(impl, "device_type", "cuda")
