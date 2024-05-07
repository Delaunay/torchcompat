"""Gaudi compatibility layer"""

import os

import torch

from torchcompat.core.errors import NotAvailable

try:
    import habana_frameworks.torch.core as htcore
except ImportError as err:
    raise NotAvailable("Could not import habana_framworks") from err


impl = htcore.hpu


ccl = "hccl"

# Not really matching the Scaler purpose though
# https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/Porting_PyTorch_Models_to_Gaudi.html
#
# class HPUStepMarker:
#     def __init__(self, enabled=True) -> None:
#         pass

#     def scale(self, loss):
#         return loss

#     def backward(self, loss):
#         loss.backward()
#         htcore.mark_step()

#     def step(self, optimizer):
#         optimizer.step()
#         htcore.mark_step()

#     def update(self):
#         pass

# if not hasattr(impl.amp, "GradScaler"):
#     setattr(impl.amp, "GradScaler", HPUStepMarker)


def init_process_group(*args, backend=None, rank=-1, world_size=-1, **kwargs):
    from habana_frameworks.torch.distributed.hccl import \
        initialize_distributed_hpu

    world_size, rank, local_rank = initialize_distributed_hpu()

    torch.distributed.init_process_group(
        *args, backend="hccl", rank=rank, world_size=world_size, **kwargs
    )


def destroy_process_group():
    torch.distributed.destroy_process_group()


# ?
os.environ["PT_HPU_LAZY_MODE"] = "0"

setattr(impl, "device_type", "hpu")
