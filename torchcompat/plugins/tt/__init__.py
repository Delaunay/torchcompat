"""Tenstorrent support via TT-XLA (PJRT + PyTorch/XLA)."""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Sequence, Tuple

from torchcompat.utils.device import Device, Event as BaseEvent, local_rank
from torchcompat.utils.errors import NotAvailable
from torchcompat.core.logs import prepare_tt_environment

from torchcompat.utils.tt_sysfs import (
    format_init_error,
    format_sysfs_summary,
    hardware_unavailable_message,
    hugepage_status,
    list_sysfs_devices,
    validate_visible_devices,
)

# Re-export sysfs helpers for ``import torchcompat.plugins.tt``.
__all__ = [
    "impl",
    "TtDevice",
    "configure_mesh",
    "format_init_error",
    "format_sysfs_summary",
    "hardware_unavailable_message",
    "hugepage_status",
    "init_mesh_group",
    "init_process_group",
    "is_data_parallel",
    "is_fsdp",
    "is_tensor_parallel",
    "list_sysfs_devices",
    "prepare_batch",
    "shard_model",
    "shard_tensor",
    "validate_visible_devices",
]

sysfs_devices = list_sysfs_devices()

if not sysfs_devices:
    raise NotAvailable(hardware_unavailable_message())

prepare_tt_environment()


class TtEvent(BaseEvent):
    def __init__(self, device: "TtDevice", **kwargs):
        super().__init__(**kwargs)
        self._device = device

    def synchronize(self):
        self._device.synchronize()


class TtDevice(Device):
    """Tenstorrent device; torch_xla is imported on first use."""

    def __init__(self):
        self._initialized = False
        self._spmd_enabled = False
        self._torch = None
        self._torch_xla = None
        self._xm = None
        self._xr = None
        self._mesh = None
        self._input_sharding_dim = None
        self._model_sharding_patterns = None
        self._param_sharding_patterns = None
        self._default_compile_options = {
            "optimization_level": "0",
            "tt_enable_torch_fx_fusion_pass": True,
            "tt_use_aot_autograd": True,
        }

    @property
    def name(self) -> str:
        return "tt"

    @property
    def device_type(self) -> str:
        return "xla"

    @property
    def ccl(self) -> str:
        return "xla"

    def _ensure_initialized(self, *, spmd: bool = False) -> None:
        """Import torch_xla and configure PJRT.

        When ``spmd=True`` (mesh / multi-chip), ``use_spmd()`` runs *before*
        any XLA device enumeration — matching blacksmith's DeviceManager order.
        """
        if self._initialized:
            if spmd and not self._spmd_enabled:
                self._setup_multichip()
            return

        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        os.environ["XLA_STABLEHLO_COMPILE"] = "1"
        pjrt = os.environ.get("PJRT_DEVICE", "").upper()
        if pjrt and pjrt != "TT":
            raise NotAvailable(f"PJRT_DEVICE is {pjrt!r}, not Tenstorrent")

        visible_error = validate_visible_devices(
            os.environ.get("TT_VISIBLE_DEVICES"),
            sysfs_devices,
        )
        if visible_error:
            raise NotAvailable(visible_error)

        os.environ.setdefault("XLA_REGISTER_INSTALLED_PLUGINS", "1")
        os.environ.setdefault("PJRT_DEVICE", "TT")

        try:
            xr.set_device_type("TT")
        except Exception as err:
            raise NotAvailable("Could not configure torch_xla for Tenstorrent") from err

        self._torch = torch
        self._torch_xla = torch_xla
        self._xm = xm
        self._xr = xr

        # SPMD must be enabled before the first XLA device/tensor is created.
        if spmd:
            self._setup_multichip()

        try:
            devices = xm.get_xla_supported_devices()
        except Exception as err:
            raise NotAvailable(format_init_error(err)) from err

        if not devices:
            raise NotAvailable("No Tenstorrent devices available")

        self._initialized = True

    def _setup_multichip(
        self, enable_trace: bool = False, trace_region_size: Optional[int] = None
    ):
        os.environ.setdefault("XLA_ALWAYS_ALLREDUCE", "1")
        os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
        os.environ.setdefault("DISABLE_NUMERIC_CC_TOKEN", "1")
        self._xr.use_spmd()
        self._spmd_enabled = True
        if enable_trace and trace_region_size is not None:
            os.environ.setdefault("TT_RUNTIME_TRACE_REGION_SIZE", str(trace_region_size))

    @property
    def Event(self):
        device = self

        class Event(TtEvent):
            def __init__(self, **kwargs):
                super().__init__(device, **kwargs)

        return Event

    def fetch_device(self, id: int | None = None):
        self._ensure_initialized()
        if id is None:
            id = local_rank()
        return self._xm.xla_device(id)

    def device_string(self, id: int | None = None) -> str:
        # PyTorch only understands the "xla" device type (not "tt").
        if id is None:
            id = local_rank()
        return f"xla:{id}"

    def synchronize(self, *args, **kwargs) -> None:
        self._ensure_initialized()
        self._torch_xla.sync(wait=True)

    def mark_step(self) -> None:
        self._ensure_initialized()
        self._xm.mark_step()

    def manual_seed(self, seed) -> None:
        self._ensure_initialized()
        self._xm.set_rng_state(int(seed))

    def manual_seed_all(self, seed) -> None:
        self._ensure_initialized()
        seed = int(seed)
        devices = self._xm.get_xla_supported_devices() or [None]
        for device in devices:
            self._xm.set_rng_state(seed, device)

    def set_device(self, device) -> None:
        """Set the current/default XLA device (cuda.set_device analog)."""
        self._ensure_initialized()
        torch_xla = self._torch_xla

        if isinstance(device, bool):
            raise TypeError(f"set_device() received an invalid device {device!r}")
        if isinstance(device, int):
            if device < 0:
                return
            torch_xla.device(device)
            return

        device_str = str(device)
        if ":" not in device_str and device_str.lstrip("-").isdigit():
            index = int(device_str)
            if index < 0:
                return
            torch_xla.device(index)
            return

        torch_xla._XLAC._xla_set_default_device(device_str)

    def device_count(self) -> int:
        # Avoid initializing XLA without SPMD when callers only need a count
        # to size ``init_mesh_group`` (must call mesh setup before any device use).
        if not self._initialized:
            visible = os.environ.get("TT_VISIBLE_DEVICES")
            if visible:
                return len([p for p in visible.split(",") if p.strip()])
            return len(sysfs_devices)
        return self._torch_xla.device_count()

    def step(self):
        self._ensure_initialized()
        return self._torch_xla.step()

    def optimizer_step(self, optimizer, barrier: bool = False, **kwargs):
        self._ensure_initialized()
        if self._mesh is None and not barrier:
            result = optimizer.step(**kwargs)
            self._torch_xla.sync(wait=True)
            return result
        return self._xm.optimizer_step(optimizer, barrier=True, **kwargs)

    def launch(self, fn, args=(), start_method="spawn", debug_single_process=False):
        self._ensure_initialized()
        return self._torch_xla.launch(
            fn,
            args=args,
            start_method=start_method,
            debug_single_process=debug_single_process,
        )

    def compile(self, model, *args, backend=None, options=None, **kwargs):
        self._ensure_initialized()
        backend = backend or "tt"
        merged = {**self._default_compile_options, **(options or {})}
        compile_kwargs = dict(kwargs)
        compile_kwargs["backend"] = backend
        compile_kwargs["options"] = merged
        return self._torch.compile(model, *args, **compile_kwargs)

    def get_mesh(self):
        return self._mesh

    def _validate_mesh_config(
        self,
        mesh_shape: Sequence[int],
        mesh_axis_names: Sequence[str],
        input_sharding_dim_arg: Optional[str] = None,
        model_sharding_patterns_arg: Optional[Sequence[Tuple]] = None,
    ):
        if mesh_axis_names is None:
            raise ValueError("mesh_axis_names is required when mesh_shape is set")
        if len(mesh_shape) != len(mesh_axis_names):
            raise ValueError("mesh_shape and mesh_axis_names must have the same length.")
        if (
            input_sharding_dim_arg is not None
            and input_sharding_dim_arg not in mesh_axis_names
        ):
            raise ValueError(
                "`input_sharding_dim` must be None or present in `mesh_axis_names`."
            )
        if model_sharding_patterns_arg is not None:
            for pattern_spec in model_sharding_patterns_arg:
                dimensions = pattern_spec[1]
                for dimension in dimensions:
                    if dimension is not None:
                        axis_index = mesh_axis_names.index(dimension)
                        if mesh_shape[axis_index] <= 1:
                            raise ValueError(
                                f"Dimension {dimension!r} has mesh size 1 for model "
                                f"sharding pattern {pattern_spec!r}."
                            )

    def _create_mesh(self, mesh_shape, axis_names):
        import torch_xla.distributed.spmd as xs

        device_ids = list(range(self._xr.global_runtime_device_count()))
        return xs.Mesh(device_ids, tuple(mesh_shape), tuple(axis_names))

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
        """Create an SPMD mesh for DP/TP/FSDP (Tenstorrent-only API).

        Must be called before any other TT device use so ``use_spmd()`` runs
        before the first XLA tensor/device is created.
        """
        self._ensure_initialized(spmd=True)
        if enable_trace:
            self._setup_multichip(
                enable_trace=enable_trace, trace_region_size=trace_region_size
            )

        num_devices = self._xr.global_runtime_device_count()

        if mesh_shape is not None:
            if mesh_axis_names is None:
                raise ValueError("mesh_axis_names is required when mesh_shape is set")
        elif auto_mesh and num_devices > 1:
            mesh_shape = (1, num_devices)
            mesh_axis_names = ("batch", "model")
        else:
            raise ValueError(
                "init_mesh_group requires mesh_shape/mesh_axis_names or auto_mesh=True"
            )

        self._validate_mesh_config(
            mesh_shape,
            mesh_axis_names,
            input_sharding_dim_arg=input_sharding_dim_arg,
            model_sharding_patterns_arg=model_sharding_patterns_arg,
        )
        self._mesh = self._create_mesh(mesh_shape, mesh_axis_names)
        self._input_sharding_dim = input_sharding_dim_arg
        self._model_sharding_patterns = model_sharding_patterns_arg
        self._param_sharding_patterns = param_sharding_patterns_arg or []
        return self._mesh

    def configure_mesh(
        self,
        input_sharding_dim_arg: Optional[str] = None,
        model_sharding_patterns_arg: Optional[Sequence[Tuple]] = None,
        param_sharding_patterns_arg: Optional[Sequence[Tuple]] = None,
        enable_trace: bool = False,
        trace_region_size: Optional[int] = None,
    ) -> None:
        self._ensure_initialized()
        if self._mesh is None:
            raise RuntimeError(
                "configure_mesh requires a mesh; call init_mesh_group first."
            )

        mesh_shape = tuple(self._mesh.shape()[name] for name in self._mesh.axis_names)
        self._validate_mesh_config(
            mesh_shape,
            self._mesh.axis_names,
            input_sharding_dim_arg=input_sharding_dim_arg,
            model_sharding_patterns_arg=model_sharding_patterns_arg,
        )
        self._setup_multichip(
            enable_trace=enable_trace, trace_region_size=trace_region_size
        )
        self._input_sharding_dim = input_sharding_dim_arg
        self._model_sharding_patterns = model_sharding_patterns_arg
        self._param_sharding_patterns = param_sharding_patterns_arg or []

    def is_data_parallel(self) -> bool:
        return (
            self._input_sharding_dim is not None
            and self._mesh is not None
            and self._mesh.shape()[self._input_sharding_dim] > 1
        )

    def is_tensor_parallel(self) -> bool:
        return self._model_sharding_patterns is not None and self._mesh is not None

    def is_fsdp(self) -> bool:
        return self._mesh is not None and "fsdp" in self._mesh.axis_names

    def shard_tensor(self, tensor, sharding_spec: Tuple):
        self._ensure_initialized()
        import torch_xla.distributed.spmd as xs

        return xs.mark_sharding(tensor, self._mesh, sharding_spec)

    def _apply_tensor_parallelism(self, model):
        import torch_xla.distributed.spmd as xs

        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue
            match = next(
                (ps for ps in self._model_sharding_patterns if re.search(ps[0], name)),
                None,
            )
            if match and self._torch_xla._XLAC._get_xla_sharding_spec(module.weight) in (
                None,
                "",
            ):
                xs.mark_sharding(module.weight, self._mesh, tuple(match[1]))

        for name, param in model.named_parameters():
            match = next(
                (ps for ps in self._param_sharding_patterns if re.search(ps[0], name)),
                None,
            )
            if match and self._torch_xla._XLAC._get_xla_sharding_spec(param) in (
                None,
                "",
            ):
                xs.mark_sharding(param, self._mesh, tuple(match[1]))

        self._torch_xla.sync(wait=True)
        return model

    def _apply_fsdp(self, model):
        import torch_xla.distributed.spmd as xs
        from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
            SpmdFullyShardedDataParallel as FSDP,
        )

        mesh = self._mesh
        torch_xla = self._torch_xla

        def shard_output(output, mesh_arg):
            real_output = getattr(output, "logits", None)
            if real_output is None:
                real_output = output[0] if isinstance(output, tuple) else output
            if torch_xla._XLAC._get_xla_sharding_spec(real_output) not in (None, ""):
                return
            partition_spec = ("fsdp",) + (None,) * (real_output.dim() - 1)
            xs.mark_sharding(real_output, mesh_arg, partition_spec)

        return FSDP(model, mesh=mesh, shard_output=shard_output)

    def shard_model(self, model):
        if self._mesh is None:
            return model
        if self.is_fsdp():
            model = self._apply_fsdp(model)
        if self.is_tensor_parallel():
            model = self._apply_tensor_parallelism(model)
        return model

    def prepare_batch(self, *args, **kwargs):
        """Move batch tensors to device and apply data-parallel sharding when configured."""
        self._ensure_initialized()
        if args and kwargs:
            raise TypeError("prepare_batch() accepts *args or **kwargs, not both")

        import torch_xla.distributed.spmd as xs

        device = self.fetch_device()

        def _prepare(value):
            import torch

            if not isinstance(value, torch.Tensor):
                return value
            value = value.to(device)
            if self.is_data_parallel() and value.dim() > 0:
                partition_spec = (self._input_sharding_dim,) + tuple(
                    [None] * (value.dim() - 1)
                )
                xs.mark_sharding(value, self._mesh, partition_spec)
            return value

        if kwargs:
            return {key: _prepare(value) for key, value in kwargs.items()}
        return tuple(_prepare(value) for value in args)

    def init_process_group(
        self, *args, backend=None, rank=-1, world_size=-1, **kwargs
    ):
        """CUDA-style process group via ``xla://`` rendezvous (not SPMD mesh)."""
        self._ensure_initialized()
        dist_kwargs = dict(kwargs)
        dist_kwargs.setdefault("init_method", "xla://")
        if rank != -1:
            dist_kwargs["rank"] = rank
        if world_size != -1:
            dist_kwargs["world_size"] = world_size
        self._torch.distributed.init_process_group(
            *args,
            backend=backend or self.ccl,
            **dist_kwargs,
        )
        return None


impl = TtDevice()

# Module-level aliases so ``from torchcompat.plugins.tt import configure_mesh``
# and lazy ``__getattr__``-style imports keep working without forcing init.
configure_mesh = impl.configure_mesh
init_mesh_group = impl.init_mesh_group
init_process_group = impl.init_process_group
is_data_parallel = impl.is_data_parallel
is_fsdp = impl.is_fsdp
is_tensor_parallel = impl.is_tensor_parallel
prepare_batch = impl.prepare_batch
shard_model = impl.shard_model
shard_tensor = impl.shard_tensor
