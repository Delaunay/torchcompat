"""Top level module for torchcompat"""

__descr__ = "torch compatibility layer"
__version__ = "1.1.4"
__license__ = "BSD 3-Clause License"
__author__ = "Anonymous"
__author_email__ = "anony@mous.com"
__copyright__ = "2024 Anonymous"
__url__ = "https://github.com/Delaunay/torchcompat"


import sys

import torch

from torchcompat.core.load import load_available

device_module = load_available()


#
# Helpers
#
def fetch_device_id():
    try:
        import os

        return int(os.getenv("LOCAL_RANK", "0"))
    except Exception:
        return 0


def device_string(id: int = fetch_device_id()):
    return f"{device_module.device_type}:{id}"


def mark_step():
    pass


def fetch_device(id: int = fetch_device_id()):
    return torch.device(device_string(id))


def init_process_group(*args, backend=None, rank=-1, world_size=-1, **kwargs):
    backend = backend or device_module.ccl
    torch.distributed.init_process_group(
        *args, backend=backend, rank=rank, world_size=world_size, **kwargs
    )


def destroy_process_group():
    torch.distributed.destroy_process_group()


#
# Default noops that gets overridden if they exist
#


def set_enable_tf32(enable=True):
    pass


def optimize(model, *args, optimizer=None, dtype=None, **kwargs):
    if dtype is not None:
        pass

    if optimizer is None:
        return model
    return model, optimizer


def empty_cache():
    pass


def synchronize():
    pass


def is_available():
    return True


class accelerate:
    def Accelerator(*args, **kwargs):
        from accelerate import Accelerator

        return Accelerator(*args, **kwargs)


#
# Add device interface to current module
#   overriding the default implementation when available
#
current_module = sys.modules[__name__]
for key, value in vars(device_module).items():
    if key.startswith("__"):
        continue
    setattr(current_module, key, value)
