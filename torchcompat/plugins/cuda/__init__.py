"""CUDA compatibility layer"""

import contextlib

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


ccl = "nccl"


@contextlib.contextmanager
def step():
    yield


def optimizer_step(optimizer, barrier=False, **kwargs):
    return optimizer.step(**kwargs)


def launch(fn, args=(), start_method="spawn", debug_single_process=False):
    fn(0, *args)


setattr(impl, "device_type", "cuda")
setattr(impl, "set_enable_tf32", set_enable_tf32)
setattr(impl, "ccl", ccl)
setattr(impl, "step", step)
setattr(impl, "optimizer_step", optimizer_step)
setattr(impl, "launch", launch)
