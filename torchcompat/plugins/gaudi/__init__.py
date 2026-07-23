"""Gaudi compatibility layer."""

from contextlib import contextmanager

import torch

from torchcompat.utils.device import Device
from torchcompat.utils.errors import NotAvailable

try:
    from habana_frameworks.torch import hpu

    hpu.init()

    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.gpu_migration  # noqa: F401
except ModuleNotFoundError as err:
    raise NotAvailable("Could not import habana_framworks") from err
except ImportError as err:
    raise NotAvailable("Could not import habana_framworks") from err

_backend = htcore.hpu

if not _backend.hpu.is_available():
    raise NotAvailable("torch.hpu is not available")


class GaudiAmp:
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


class GaudiAccelerate:
    @staticmethod
    def Accelerator(*args, **kwargs):
        from optimum.habana.accelerate.accelerator import GaudiAccelerator

        class Custom(GaudiAccelerator):
            def backward(self, *args, **kwargs):
                super().backward(*args, **kwargs)
                htcore.mark_step()

        return Custom(*args, **kwargs)


class GaudiDevice(Device):
    def __init__(self):
        self._backend = _backend

    def __getattr__(self, name: str):
        try:
            return getattr(self._backend, name)
        except AttributeError as err:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from err

    @property
    def name(self) -> str:
        return "gaudi"

    @property
    def device_type(self) -> str:
        return "hpu"

    @property
    def ccl(self) -> str:
        return "hccl"

    def device_string(self, id: int | None = None) -> str:
        if id is None:
            from torchcompat.utils.device import local_rank

            id = local_rank()
        return f"hpu:{id}"

    def fetch_device(self, id: int | None = None):
        return torch.device("hpu", torch.hpu.current_device())

    def synchronize(self, *args, **kwargs) -> None:
        # make a step to force compute
        htcore.mark_step()
        # Default synchronize does not sync
        self._backend.default_stream().synchronize()

    def set_enable_tf32(self, enable: bool = True) -> None:
        print("HPU cannot disable tf32")

    def mark_step(self) -> None:
        htcore.mark_step()

    def init_process_group(
        self, *args, backend=None, rank=-1, world_size=-1, **kwargs
    ):
        import habana_frameworks.torch.distributed.hccl  # noqa: F401
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

        print(world_size, rank, kwargs)
        world_size, rank, local_rank = initialize_distributed_hpu()

        print(world_size, rank, local_rank)
        torch.distributed.init_process_group(
            *args, backend="hccl", rank=rank, world_size=world_size, **kwargs
        )

    def optimizer_step(self, optimizer, barrier: bool = False, **kwargs):
        result = optimizer.step(**kwargs)
        self.mark_step()
        return result

    @property
    def amp(self):
        return GaudiAmp

    @property
    def accelerate(self):
        return GaudiAccelerate


impl = GaudiDevice()
