"""ROCm compatibility layer"""

import torch
import os

from torchcompat.core.errors import NotAvailable

if not torch.cuda.is_available():
    raise NotAvailable("torch.cuda is not available")

# check that torch.cuda is in fact rocm
if not torch.version.hip:
    raise NotAvailable("torch.cuda is not rocm")


def set_enable_tf32(enable=True):
    if enable:
        os.environ["HIPBLASLT_ALLOW_TF32"] = "1"
    else:
        os.environ["HIPBLASLT_ALLOW_TF32"] = "0"
        
    torch.backends.cuda.matmul.allow_tf32 = enable
    torch.backends.cudnn.allow_tf32 = enable


impl = torch.cuda

ccl = "nccl"

setattr(impl, "device_type", "cuda")
setattr(impl, "ccl", ccl)
setattr(impl, "set_enable_tf32", set_enable_tf32)
