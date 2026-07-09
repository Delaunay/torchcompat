"""Plugin example"""

import contextlib
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


ccl = "gloo"


@contextlib.contextmanager
def step():
    yield


def optimizer_step(optimizer, barrier=False, **kwargs):
    return optimizer.step(**kwargs)


def launch(fn, args=(), start_method="spawn", debug_single_process=False):
    fn(0, *args)


setattr(impl, "device_type", "cpu")
setattr(impl, "set_enable_tf32", set_enable_tf32)
setattr(impl, "ccl", ccl)
setattr(impl, "Event", Event)
setattr(impl, "step", step)
setattr(impl, "optimizer_step", optimizer_step)
setattr(impl, "launch", launch)
