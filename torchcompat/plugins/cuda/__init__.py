"""CUDA compatibility layer"""

import torch

from torchcompat.core.errors import NotAvailable

if not torch.cuda.is_available():
    raise NotAvailable("torch.cuda is not available")

# check that torch.cuda is in fact cuda and NOT rocm
if not torch.version.cuda:
    raise NotAvailable("torch.cuda is not rocm")


impl = torch.cuda


def set_enable_tf32(enable=True):
    torch.backends.cuda.matmul.allow_tf32 = enable
    torch.backends.cudnn.allow_tf32 = enable


def compile(model, **kwargs):
    return torch.compile(model, **kwargs)

ccl = "nccl"

compile_backend = None

setattr(impl, "compile_backend", compile_backend)
setattr(impl, "compile", compile)
setattr(impl, "device_type", "cuda")
setattr(impl, "set_enable_tf32", set_enable_tf32)
setattr(impl, "ccl", ccl)
