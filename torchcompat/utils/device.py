"""Generic device interface implemented by each torchcompat plugin."""

from __future__ import annotations

import contextlib
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple


def local_rank() -> int:
    try:
        return int(os.getenv("LOCAL_RANK", "0"))
    except Exception:
        return 0


class Event:
    """Fallback timing event used when the backend has no native Event."""

    def __init__(self, **kwargs):
        self.start = 0

    def record(self):
        self.synchronize()
        self.start = time.time()

    def elapsed_time(self, end: "Event"):
        return (end.start - self.start) * 1000

    def synchronize(self):
        pass


class Device(ABC):
    """Unified accelerator interface.

    Plugins subclass this and export ``impl = SomeDevice()``. Optional
    TT/XLA helpers have safe defaults so CUDA/CPU backends stay thin.
    """

    # Attributes / methods flattened onto ``torchcompat.core``.
    PUBLIC_ATTRS: Tuple[str, ...] = (
        "name",
        "device_type",
        "ccl",
        "Event",
        "amp",
        "is_available",
        "device_count",
        "fetch_device",
        "device_string",
        "set_device",
        "synchronize",
        "empty_cache",
        "manual_seed",
        "manual_seed_all",
        "set_enable_tf32",
        "step",
        "optimizer_step",
        "launch",
        "compile",
        "init_process_group",
        "init_mesh_group",
        "destroy_process_group",
        "mark_step",
        "get_mesh",
        "configure_mesh",
        "is_data_parallel",
        "is_tensor_parallel",
        "is_fsdp",
        "shard_tensor",
        "shard_model",
        "prepare_batch",
        "accelerate",
    )

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin identity (``cuda``, ``tt``, ``rocm``, ...)."""

    @property
    @abstractmethod
    def device_type(self) -> str:
        """PyTorch device type string (``cuda``, ``xla``, ``hpu``, ...)."""

    @property
    @abstractmethod
    def ccl(self) -> str:
        """Default collective backend name for ``init_process_group``."""

    # --- core ---

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def device_string(self, id: int | None = None) -> str:
        if id is None:
            id = local_rank()
        return f"{self.device_type}:{id}"

    def fetch_device(self, id: int | None = None):
        import torch

        if id is None:
            id = local_rank()
        return torch.device(self.device_string(id))

    def set_device(self, device) -> None:
        pass

    def synchronize(self, *args, **kwargs) -> None:
        pass

    def empty_cache(self) -> None:
        pass

    def manual_seed(self, seed) -> None:
        import torch

        torch.manual_seed(int(seed))

    def manual_seed_all(self, seed) -> None:
        self.manual_seed(seed)

    def set_enable_tf32(self, enable: bool = True) -> None:
        pass

    @property
    def Event(self):
        return Event

    @contextlib.contextmanager
    def step(self):
        yield

    def optimizer_step(self, optimizer, barrier: bool = False, **kwargs):
        result = optimizer.step(**kwargs)
        # Lazy backends (XLA/TT/Gaudi) flush via ``mark_step``; no-op elsewhere.
        self.mark_step()
        return result

    def launch(self, fn, args=(), start_method="spawn", debug_single_process=False):
        fn(0, *args)

    def compile(self, model, *args, **kwargs):
        import torch

        return torch.compile(model, *args, **kwargs)

    def init_process_group(
        self, *args, backend=None, rank=-1, world_size=-1, **kwargs
    ):
        """CUDA-style ``torch.distributed`` process group (NCCL/gloo/…)."""
        import torch

        dist_kwargs = dict(kwargs)
        if rank != -1:
            dist_kwargs["rank"] = rank
        if world_size != -1:
            dist_kwargs["world_size"] = world_size
        torch.distributed.init_process_group(
            *args, backend=backend or self.ccl, **dist_kwargs
        )
        return None

    def init_mesh_group(
        self,
        mesh_shape=None,
        mesh_axis_names=None,
        *,
        auto_mesh: bool = False,
        input_sharding_dim_arg: Optional[str] = None,
        model_sharding_patterns_arg: Optional[Sequence[Tuple]] = None,
        param_sharding_patterns_arg: Optional[Sequence[Tuple]] = None,
        enable_trace: bool = False,
        trace_region_size: Optional[int] = None,
    ):
        """SPMD mesh setup (Tenstorrent). No-op on other backends."""
        return None

    def destroy_process_group(self) -> None:
        import torch

        torch.distributed.destroy_process_group()

    @property
    def amp(self):
        backend_device_type = self.device_type

        class amp:
            @staticmethod
            def autocast(*args, device_type=None, **kwargs):
                import torch

                if not args and device_type is None:
                    device_type = backend_device_type
                if device_type is not None:
                    return torch.amp.autocast(device_type, *args, **kwargs)
                return torch.amp.autocast(*args, **kwargs)

            @staticmethod
            def GradScaler(*args, device=None, **kwargs):
                import torch

                if device is None:
                    device = backend_device_type
                return torch.amp.GradScaler(*args, device=device, **kwargs)

        return amp

    @property
    def accelerate(self):
        class accelerate:
            @staticmethod
            def Accelerator(*args, **kwargs):
                from accelerate import Accelerator

                return Accelerator(*args, **kwargs)

        return accelerate

    # --- extras (TT / XLA override) ---

    def mark_step(self) -> None:
        pass

    def get_mesh(self):
        return None

    def configure_mesh(
        self,
        input_sharding_dim_arg: Optional[str] = None,
        model_sharding_patterns_arg: Optional[Sequence[Tuple]] = None,
        param_sharding_patterns_arg: Optional[Sequence[Tuple]] = None,
        enable_trace: bool = False,
        trace_region_size: Optional[int] = None,
    ) -> None:
        if self.get_mesh() is None and any(
            value is not None
            for value in (
                input_sharding_dim_arg,
                model_sharding_patterns_arg,
                param_sharding_patterns_arg,
            )
        ):
            raise RuntimeError(
                "configure_mesh requires a mesh; call init_mesh_group first."
            )

    def is_data_parallel(self) -> bool:
        return False

    def is_tensor_parallel(self) -> bool:
        return False

    def is_fsdp(self) -> bool:
        return False

    def shard_tensor(self, tensor, sharding_spec: Tuple):
        return tensor

    def shard_model(self, model):
        return model

    def prepare_batch(self, *args, **kwargs):
        """Move batch tensors to this device (and shard on SPMD backends).

        Call with either positional or keyword tensors — not both::

            x, y = accelerator.prepare_batch(x, y)
            batch = accelerator.prepare_batch(images=x, labels=y)

        Returns a tuple for ``*args`` and a dict for ``**kwargs``.
        """
        if args and kwargs:
            raise TypeError("prepare_batch() accepts *args or **kwargs, not both")

        device = self.fetch_device()

        def _prepare(value):
            import torch

            if isinstance(value, torch.Tensor):
                return value.to(device)
            return value

        if kwargs:
            return {key: _prepare(value) for key, value in kwargs.items()}
        return tuple(_prepare(value) for value in args)


class TorchBackendDevice(Device):
    """Device that wraps a ``torch.<backend>`` module (cuda, xpu, cpu, ...)."""

    def __init__(self, backend):
        self._backend = backend

    def __getattr__(self, name: str):
        # Allow ``accelerator.max_memory_allocated`` etc. via core flattening
        # / module ``__getattr__`` forwarding.
        try:
            return getattr(self._backend, name)
        except AttributeError as err:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from err

    def is_available(self) -> bool:
        return bool(self._backend.is_available())

    def device_count(self) -> int:
        return int(self._backend.device_count())

    def set_device(self, device) -> None:
        self._backend.set_device(device)

    def synchronize(self, *args, **kwargs) -> None:
        self._backend.synchronize(*args, **kwargs)

    def empty_cache(self) -> None:
        if hasattr(self._backend, "empty_cache"):
            self._backend.empty_cache()

    def manual_seed(self, seed) -> None:
        if hasattr(self._backend, "manual_seed"):
            self._backend.manual_seed(int(seed))
        else:
            super().manual_seed(seed)

    def manual_seed_all(self, seed) -> None:
        if hasattr(self._backend, "manual_seed_all"):
            self._backend.manual_seed_all(int(seed))
        else:
            super().manual_seed_all(seed)

    @property
    def Event(self):
        if hasattr(self._backend, "Event"):
            return self._backend.Event
        return Event

    @property
    def amp(self):
        # Prefer ``torch.amp``; ``torch.cuda.amp`` / backend ``.amp`` are deprecated
        # but kept as a fallback for older PyTorch installs.
        import torch

        if hasattr(torch, "amp"):
            return super().amp
        if hasattr(self._backend, "amp"):
            return self._backend.amp
        return super().amp
