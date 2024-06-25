"""Gaudi compatibility layer"""

import os
from contextlib import contextmanager

import torch

from torchcompat.core.errors import NotAvailable

try:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.gpu_migration
except ModuleNotFoundError as err:
    raise NotAvailable("Could not import habana_framworks") from err
except ImportError as err:
    raise NotAvailable("Could not import habana_framworks") from err


impl = htcore.hpu

if not impl.hpu.is_available():
    raise NotAvailable("torch.hpu is not available")

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
    import habana_frameworks.torch.distributed.hccl
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

    world_size, rank, local_rank = initialize_distributed_hpu()

    torch.distributed.init_process_group(
        *args, backend="hccl", rank=rank, world_size=world_size, **kwargs
    )


def destroy_process_group():
    torch.distributed.destroy_process_group()


def realsynch():
    # make a step to force compute
    htcore.mark_step()

    # Default synchronize does not sync
    impl.default_stream().synchronize()


def set_enable_tf32(enable=True):
    print("HPU cannot disable tf32")


class amp:
    @contextmanager
    @staticmethod
    def autocast(*args, device_type=None, **kwargs):
        device_type = "hpu"
        with torch.autocast(*args, device_type=device_type, **kwargs):
            yield

    class GradScaler:
        def __init__(self, *args, enabled=False, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def scale(self, *args):
            if len(args) == 1:
                return args[0]
            return args

        def step(self, optimizer, *args, **kwargs):
            return optimizer.step(*args, **kwargs)

        def update(self, *args):
            pass

    # class GradScaler(torch.amp.GradScaler):
    #     def __init__(
    #         self,
    #         init_scale: float = 2.0**16,
    #         growth_factor: float = 2.0,
    #         backoff_factor: float = 0.5,
    #         growth_interval: int = 2000,
    #         enabled: bool = True,
    #     ) -> None:
    #         super().__init__(
    #             "hpu",
    #             init_scale=init_scale,
    #             growth_factor=growth_factor,
    #             backoff_factor=backoff_factor,
    #             growth_interval=growth_interval,
    #             enabled=enabled,
    #         )


def device_string(id: int):
    return "hpu"


#
# Huggingface
#
class accelerate:
    def Accelerator(*args, **kwargs):
        from optimum.habana.accelerate.accelerator import GaudiAccelerator

        return GaudiAccelerator(*args, **kwargs)


setattr(impl, "accelerate", accelerate)

setattr(impl, "device_string", device_string)
setattr(impl, "device_type", "hpu")
setattr(impl, "synchronize", realsynch)
setattr(impl, "set_enable_tf32", set_enable_tf32)
setattr(impl, "mark_step", htcore.mark_step)
setattr(impl, "amp", amp)
setattr(impl, "ccl", ccl)
