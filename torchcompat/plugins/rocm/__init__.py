"""ROCm compatibility layer"""

import torch

from torchcompat.core.errors import NotAvailable

if not torch.cuda.is_available():
    raise NotAvailable("torch.cuda is not available")

# check that torch.cuda is in fact rocm
if not torch.version.hip:
    raise NotAvailable("torch.cuda is not rocm")


impl = torch.cuda

ccl = "nccl"

setattr(impl, "device_type", "cuda")
setattr(impl, "ccl", ccl)
