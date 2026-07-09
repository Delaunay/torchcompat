"""Tenstorrent support via TT-XLA (PJRT + PyTorch/XLA)."""

from __future__ import annotations

import os
import re
import types
from typing import Dict, Optional, Sequence, Tuple

from torchcompat.core.errors import NotAvailable
from torchcompat.core.logs import log_root

from .sysfs import (
    format_init_error,
    format_sysfs_summary,
    hardware_unavailable_message,
    hugepage_status,
    list_sysfs_devices,
    validate_visible_devices,
)

_INITIALIZED = False

# Re-export sysfs helpers for ``import torchcompat.plugins.tt``.
__all__ = [
    "impl",
    "configure_mesh",
    "format_init_error",
    "format_sysfs_summary",
    "hardware_unavailable_message",
    "hugepage_status",
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

_RUNTIME_NAMES = frozenset(
    {
        "impl",
        "ccl",
        "fetch_device",
        "device_string",
        "set_enable_tf32",
        "synchronize",
        "mark_step",
        "optimizer_step",
        "compile",
        "get_mesh",
        "configure_mesh",
        "is_data_parallel",
        "is_tensor_parallel",
        "is_fsdp",
        "shard_tensor",
        "shard_model",
        "prepare_batch",
        "init_process_group",
        "_init_process_group",
    }
)



sysfs_devices = list_sysfs_devices()

if not sysfs_devices:
    raise NotAvailable(hardware_unavailable_message())

LOG_PATH = log_root() / "tt" / "logger.txt"
os.environ["TT_LOGGER_FILE"] = str(LOG_PATH)


def __getattr__(name: str):
    if name in _RUNTIME_NAMES:

        _ensure_initialized()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _ensure_initialized() -> None:
    global _INITIALIZED

    if _INITIALIZED:
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

    try:
        devices = xm.get_xla_supported_devices()
    except Exception as err:
        raise NotAvailable(format_init_error(err)) from err

    if not devices:
        raise NotAvailable("No Tenstorrent devices available")

    mesh = None
    input_sharding_dim = None
    model_sharding_patterns = None
    param_sharding_patterns = None
    ccl = "xla"

    default_compile_options = {
        "optimization_level": "0",
        "tt_enable_torch_fx_fusion_pass": True,
        "tt_use_aot_autograd": True,
    }

    def fetch_device(id: int = 0):
        return xm.xla_device(id)

    def device_string(id: int = 0):
        return f"tt:{id}"

    def set_enable_tf32(enable=True):
        pass

    def synchronize():
        torch_xla.sync(wait=True)

    def mark_step():
        xm.mark_step()

    def optimizer_step(optimizer, barrier=False, **kwargs):
        if mesh is None and not barrier:
            result = optimizer.step(**kwargs)
            torch_xla.sync(wait=True)
            return result
        return xm.optimizer_step(optimizer, barrier=True, **kwargs)

    def compile(model, *args, backend=None, options=None, **kwargs):
        backend = backend or "tt"
        merged = {**default_compile_options, **(options or {})}
        compile_kwargs = dict(kwargs)
        compile_kwargs["backend"] = backend
        compile_kwargs["options"] = merged
        return torch.compile(model, *args, **compile_kwargs)

    def get_mesh():
        return mesh

    def _validate_mesh_config(
        mesh_shape: Sequence[int],
        mesh_axis_names: Sequence[str],
        input_sharding_dim_arg: Optional[str] = None,
        model_sharding_patterns_arg: Optional[Sequence[Tuple]] = None,
    ):
        if mesh_axis_names is None:
            raise ValueError("mesh_axis_names is required when mesh_shape is set")
        if len(mesh_shape) != len(mesh_axis_names):
            raise ValueError("mesh_shape and mesh_axis_names must have the same length.")
        if input_sharding_dim_arg is not None and input_sharding_dim_arg not in mesh_axis_names:
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

    def _setup_multichip(enable_trace: bool = False, trace_region_size: Optional[int] = None):
        os.environ.setdefault("XLA_ALWAYS_ALLREDUCE", "1")
        os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
        os.environ.setdefault("DISABLE_NUMERIC_CC_TOKEN", "1")
        xr.use_spmd()
        if enable_trace and trace_region_size is not None:
            os.environ.setdefault("TT_RUNTIME_TRACE_REGION_SIZE", str(trace_region_size))

    def _create_mesh(mesh_shape, axis_names):
        import torch_xla.distributed.spmd as xs

        device_ids = list(range(xr.global_runtime_device_count()))
        return xs.Mesh(device_ids, tuple(mesh_shape), tuple(axis_names))

    def _init_process_group(
        *args,
        backend=None,
        rank=-1,
        world_size=-1,
        mesh_shape=None,
        mesh_axis_names=None,
        auto_mesh=False,
        **kwargs,
    ):
        nonlocal mesh

        num_devices = xr.global_runtime_device_count()
        if mesh_shape is not None:
            if mesh_axis_names is None:
                raise ValueError("mesh_axis_names is required when mesh_shape is set")
        elif auto_mesh and num_devices > 1:
            mesh_shape = (1, num_devices)
            mesh_axis_names = ("batch", "model")

        if mesh_shape is not None:
            _setup_multichip()
            mesh = _create_mesh(mesh_shape, mesh_axis_names)
            return mesh

        dist_kwargs = dict(kwargs)
        dist_kwargs.setdefault("init_method", "xla://")
        if rank != -1:
            dist_kwargs["rank"] = rank
        if world_size != -1:
            dist_kwargs["world_size"] = world_size
        torch.distributed.init_process_group(
            *args,
            backend=backend or ccl,
            **dist_kwargs,
        )
        return None

    def configure_mesh(
        input_sharding_dim_arg: Optional[str] = None,
        model_sharding_patterns_arg: Optional[Sequence[Tuple]] = None,
        param_sharding_patterns_arg: Optional[Sequence[Tuple]] = None,
        enable_trace: bool = False,
        trace_region_size: Optional[int] = None,
    ):
        nonlocal input_sharding_dim, model_sharding_patterns, param_sharding_patterns

        if mesh is None:
            raise RuntimeError("configure_mesh requires a mesh; call init_process_group first.")

        mesh_shape = tuple(mesh.shape()[name] for name in mesh.axis_names)
        _validate_mesh_config(
            mesh_shape,
            mesh.axis_names,
            input_sharding_dim_arg=input_sharding_dim_arg,
            model_sharding_patterns_arg=model_sharding_patterns_arg,
        )
        _setup_multichip(enable_trace=enable_trace, trace_region_size=trace_region_size)
        input_sharding_dim = input_sharding_dim_arg
        model_sharding_patterns = model_sharding_patterns_arg
        param_sharding_patterns = param_sharding_patterns_arg or []

    def is_data_parallel() -> bool:
        return (
            input_sharding_dim is not None
            and mesh is not None
            and mesh.shape()[input_sharding_dim] > 1
        )

    def is_tensor_parallel() -> bool:
        return model_sharding_patterns is not None and mesh is not None

    def is_fsdp() -> bool:
        return mesh is not None and "fsdp" in mesh.axis_names

    def shard_tensor(tensor: torch.Tensor, sharding_spec: Tuple):
        import torch_xla.distributed.spmd as xs

        return xs.mark_sharding(tensor, mesh, sharding_spec)

    def _apply_tensor_parallelism(model: torch.nn.Module) -> torch.nn.Module:
        import torch_xla.distributed.spmd as xs

        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue
            match = next(
                (ps for ps in model_sharding_patterns if re.search(ps[0], name)),
                None,
            )
            if match and torch_xla._XLAC._get_xla_sharding_spec(module.weight) in (None, ""):
                xs.mark_sharding(module.weight, mesh, tuple(match[1]))

        for name, param in model.named_parameters():
            match = next(
                (ps for ps in param_sharding_patterns if re.search(ps[0], name)),
                None,
            )
            if match and torch_xla._XLAC._get_xla_sharding_spec(param) in (None, ""):
                xs.mark_sharding(param, mesh, tuple(match[1]))

        torch_xla.sync(wait=True)
        return model

    def _apply_fsdp(model: torch.nn.Module) -> torch.nn.Module:
        import torch_xla.distributed.spmd as xs
        from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
            SpmdFullyShardedDataParallel as FSDP,
        )

        def shard_output(output, mesh_arg):
            real_output = getattr(output, "logits", None)
            if real_output is None:
                real_output = output[0] if isinstance(output, tuple) else output
            if torch_xla._XLAC._get_xla_sharding_spec(real_output) not in (None, ""):
                return
            partition_spec = ("fsdp",) + (None,) * (real_output.dim() - 1)
            xs.mark_sharding(real_output, mesh_arg, partition_spec)

        return FSDP(model, mesh=mesh, shard_output=shard_output)

    def shard_model(model: torch.nn.Module) -> torch.nn.Module:
        if mesh is None:
            return model
        if is_fsdp():
            model = _apply_fsdp(model)
        if is_tensor_parallel():
            model = _apply_tensor_parallelism(model)
        return model

    def prepare_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        import torch_xla.distributed.spmd as xs

        device = fetch_device()
        batch = {k: v.to(device) for k, v in batch.items()}
        if is_data_parallel():
            for _, tensor in batch.items():
                if tensor.dim() > 0:
                    partition_spec = (input_sharding_dim,) + tuple([None] * (tensor.dim() - 1))
                    xs.mark_sharding(tensor, mesh, partition_spec)
        return batch

    def init_process_group(
        *args,
        backend=None,
        rank=-1,
        world_size=-1,
        mesh_shape=None,
        mesh_axis_names=None,
        input_sharding_dim_arg=None,
        model_sharding_patterns_arg=None,
        param_sharding_patterns_arg=None,
        enable_trace=False,
        trace_region_size=None,
        auto_mesh=False,
        **kwargs,
    ):
        nonlocal input_sharding_dim, model_sharding_patterns, param_sharding_patterns

        if mesh_shape is not None:
            _validate_mesh_config(
                mesh_shape,
                mesh_axis_names,
                input_sharding_dim_arg=input_sharding_dim_arg,
                model_sharding_patterns_arg=model_sharding_patterns_arg,
            )

        mesh_result = _init_process_group(
            *args,
            backend=backend,
            rank=rank,
            world_size=world_size,
            mesh_shape=mesh_shape,
            mesh_axis_names=mesh_axis_names,
            auto_mesh=auto_mesh,
            **kwargs,
        )
        if mesh_result is not None and any(
            value is not None
            for value in (
                input_sharding_dim_arg,
                model_sharding_patterns_arg,
                param_sharding_patterns_arg,
                enable_trace,
                trace_region_size,
            )
        ):
            configure_mesh(
                input_sharding_dim_arg=input_sharding_dim_arg,
                model_sharding_patterns_arg=model_sharding_patterns_arg,
                param_sharding_patterns_arg=param_sharding_patterns_arg,
                enable_trace=enable_trace,
                trace_region_size=trace_region_size,
            )
        elif mesh_result is None:
            input_sharding_dim = None
            model_sharding_patterns = None
            param_sharding_patterns = None
        return mesh_result

    impl = types.SimpleNamespace()
    setattr(impl, "device_type", "tt")
    setattr(impl, "ccl", ccl)
    setattr(impl, "set_enable_tf32", set_enable_tf32)
    setattr(impl, "fetch_device", fetch_device)
    setattr(impl, "device_string", device_string)
    setattr(impl, "synchronize", synchronize)
    setattr(impl, "mark_step", mark_step)
    setattr(impl, "step", torch_xla.step)
    setattr(impl, "optimizer_step", optimizer_step)
    setattr(impl, "launch", torch_xla.launch)
    setattr(impl, "compile", compile)
    setattr(impl, "get_mesh", get_mesh)
    setattr(impl, "init_process_group", _init_process_group)

    globals().update(
        {
            "impl": impl,
            "ccl": ccl,
            "fetch_device": fetch_device,
            "device_string": device_string,
            "set_enable_tf32": set_enable_tf32,
            "synchronize": synchronize,
            "mark_step": mark_step,
            "optimizer_step": optimizer_step,
            "compile": compile,
            "get_mesh": get_mesh,
            "configure_mesh": configure_mesh,
            "is_data_parallel": is_data_parallel,
            "is_tensor_parallel": is_tensor_parallel,
            "is_fsdp": is_fsdp,
            "shard_tensor": shard_tensor,
            "shard_model": shard_model,
            "prepare_batch": prepare_batch,
            "init_process_group": init_process_group,
            "_init_process_group": _init_process_group,
        }
    )
    _INITIALIZED = True
