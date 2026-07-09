"""XLA support for PyTorch (TPU, CPU, CUDA, Neuron, etc.)"""

import os
import types

import torch

from torchcompat.core.errors import NotAvailable

if os.environ.get("PJRT_DEVICE", "").upper() == "TT":
    raise NotAvailable("Tenstorrent devices use the TT plugin")

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
except ImportError as err:
    raise NotAvailable("Could not import torch_xla") from err

try:
    _devices = torch_xla.devices()
except Exception as err:
    raise NotAvailable("torch_xla could not enumerate devices") from err

if not _devices:
    raise NotAvailable("No XLA devices available")


impl = types.SimpleNamespace()
_mesh = None
ccl = "xla"


def fetch_device(id: int = 0):
    return torch_xla.device(id)


def device_string(id: int = 0):
    return f"xla:{id}"


def set_enable_tf32(enable=True):
    pass


def synchronize():
    torch_xla.sync(wait=True)


def mark_step():
    torch_xla.sync()


def optimizer_step(optimizer, barrier=False, **kwargs):
    return xm.optimizer_step(optimizer, barrier=barrier, **kwargs)


def compile(model, *args, backend=None, options=None, **kwargs):
    compile_kwargs = dict(kwargs)
    if options is not None:
        torch_xla.set_custom_compile_options(options)
        compile_kwargs["custom_compile_options"] = options
    return torch_xla.compile(model, *args, **compile_kwargs)


def get_mesh():
    return _mesh


def _setup_multichip():
    xr.use_spmd()


def _create_mesh(mesh_shape, axis_names):
    import torch_xla.distributed.spmd as xs

    device_ids = list(range(xr.global_runtime_device_count()))
    return xs.Mesh(device_ids, tuple(mesh_shape), tuple(axis_names))


def init_process_group(
    *args,
    backend=None,
    rank=-1,
    world_size=-1,
    mesh_shape=None,
    mesh_axis_names=None,
    auto_mesh=False,
    **kwargs,
):
    global _mesh

    num_devices = xr.global_runtime_device_count()
    if mesh_shape is not None:
        if mesh_axis_names is None:
            raise ValueError("mesh_axis_names is required when mesh_shape is set")
    elif auto_mesh and num_devices > 1:
        mesh_shape = (1, num_devices)
        mesh_axis_names = ("batch", "model")

    if mesh_shape is not None:
        _setup_multichip()
        _mesh = _create_mesh(mesh_shape, mesh_axis_names)
        return _mesh

    # xla:// is registered by torch_xla as a custom rendezvous URL scheme.
    # See torch_xla.distributed.xla_backend and _internal/rendezvous.py.
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


setattr(impl, "device_type", "xla")
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
setattr(impl, "init_process_group", init_process_group)
