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
compile_backend = None


def compile(model, backend=None, **kwargs):
    return torch.compile(model, backend=backend, **kwargs)

setattr(impl, "compile_backend", compile_backend)
setattr(impl, "compile", compile)
setattr(impl, "device_type", "cuda")
setattr(impl, "ccl", ccl)
