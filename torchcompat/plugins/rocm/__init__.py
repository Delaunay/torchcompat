"""ROCm compatibility layer"""

import contextlib

import torch

from torchcompat.core.errors import NotAvailable

if not torch.cuda.is_available():
    raise NotAvailable("torch.cuda is not available")

# check that torch.cuda is in fact rocm
if not torch.version.hip:
    raise NotAvailable("torch.cuda is not rocm")


impl = torch.cuda

ccl = "nccl"


@contextlib.contextmanager
def step():
    yield


def optimizer_step(optimizer, barrier=False, **kwargs):
    return optimizer.step(**kwargs)


def launch(fn, args=(), start_method="spawn", debug_single_process=False):
    fn(0, *args)


setattr(impl, "device_type", "cuda")
setattr(impl, "ccl", ccl)
setattr(impl, "step", step)
setattr(impl, "optimizer_step", optimizer_step)
setattr(impl, "launch", launch)
