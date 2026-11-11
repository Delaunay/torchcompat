"""Plugin example"""

import time

import torch

from torchcompat.core.errors import NotAvailable

impl = torch.cpu


def set_enable_tf32(enable=True):
    pass


class Event:
    def __init__(self, **kwargs):
        self.start = 0

    def record(self):
        self.start = time.time()

    def elapsed_time(self, end):
        # should return ms
        return (end.start - self.start) * 1000

    def synchronize(self):
        pass

compile_backend = None

def compile(model, backend=None, **kwargs):
    return torch.compile(model, backend=backend, **kwargs)

ccl = "gloo"
setattr(impl, "compile_backend", compile_backend)
setattr(impl, "compile", compile)
setattr(impl, "device_type", "cpu")
setattr(impl, "set_enable_tf32", set_enable_tf32)
setattr(impl, "ccl", ccl)
setattr(impl, "Event", Event)
