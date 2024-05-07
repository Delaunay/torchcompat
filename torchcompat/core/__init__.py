"""Top level module for torchcompat"""


__descr__ = "torch compatibility layer"
__version__ = "0.0.1"
__license__ = "BSD 3-Clause License"
__author__ = "Anonymous"
__author_email__ = "anony@mous.com"
__copyright__ = "2024 Anonymous"
__url__ = "https://github.com/Delaunay/torchcompat"


import sys

from torchcompat.core.load import load_device

import torch

device_module = load_device()


#
# Helpers
#
def device_string(id: int):
    return f"{device_module.device_name}:{id}"


def fetch_device(id: int):
    return torch.device(device_string(id))


def init_process_group(*args, backend=None, rank=-1, world_size=-1, **kwargs):
    backend = backend or device_module.ccl
    torch.distributed.init_process_group(*args, backend=backend, rank=rank, world_size=world_size, **kwargs)


def destroy_process_group():
    torch.distributed.destroy_process_group()

#
# Default noops that gets overriden if they exist
#

# Not all device support tf32
def set_enable_tf32(enable=True):
    pass

#
# XPU has a special optimizer
#
def optimizer(model, *args, optimizer=None, dtype=None, **kwargs):
    if dtype is not None:
        # model.to(dtype=dtype) ?
        pass

    if optimizer is None:
        return model
    else:
        return model, optimizer


#
# This actually cannot really trigger because  load_device would raise NoDeviceDetected
# so this does not make it possible to fallback on CPU
def is_available():
    return True

#
# Add device interface to current module
#   overriding the default implementation when available
#
self = current_module = sys.modules[__name__]
for k, v in vars(device_module).items():
    setattr(self, k, v)
