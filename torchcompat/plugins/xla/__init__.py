"""XLA support for PyTorch (TPU, CPU, CUDA, Neuron, etc.)."""

from __future__ import annotations

import os

import torch

from torchcompat.utils.device import Device
from torchcompat.utils.errors import NotAvailable

if os.environ.get("PJRT_DEVICE", "").upper() == "TT":
    raise NotAvailable("Tenstorrent devices use the TT plugin")

from torchcompat.utils.tt_sysfs import list_sysfs_devices

if list_sysfs_devices():
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


class XlaDevice(Device):
    def __init__(self):
        self._mesh = None

    @property
    def name(self) -> str:
        return "xla"

    @property
    def device_type(self) -> str:
        return "xla"

    @property
    def ccl(self) -> str:
        return "xla"

    def fetch_device(self, id: int | None = None):
        if id is None:
            from torchcompat.utils.device import local_rank

            id = local_rank()
        return torch_xla.device(id)

    def device_string(self, id: int | None = None) -> str:
        if id is None:
            from torchcompat.utils.device import local_rank

            id = local_rank()
        return f"xla:{id}"

    def synchronize(self, *args, **kwargs) -> None:
        torch_xla.sync(wait=True)

    def mark_step(self) -> None:
        torch_xla.sync()

    def manual_seed(self, seed) -> None:
        xm.set_rng_state(int(seed))

    def manual_seed_all(self, seed) -> None:
        seed = int(seed)
        devices = xm.get_xla_supported_devices() or [None]
        for device in devices:
            xm.set_rng_state(seed, device)

    def device_count(self) -> int:
        return torch_xla.device_count()

    def step(self):
        return torch_xla.step()

    def optimizer_step(self, optimizer, barrier: bool = False, **kwargs):
        result = xm.optimizer_step(optimizer, barrier=barrier, **kwargs)
        if not barrier:
            self.mark_step()
        return result

    def launch(self, fn, args=(), start_method="spawn", debug_single_process=False):
        return torch_xla.launch(
            fn,
            args=args,
            start_method=start_method,
            debug_single_process=debug_single_process,
        )

    def compile(self, model, *args, backend=None, options=None, **kwargs):
        compile_kwargs = dict(kwargs)
        if options is not None:
            torch_xla.set_custom_compile_options(options)
            compile_kwargs["custom_compile_options"] = options
        return torch_xla.compile(model, *args, **compile_kwargs)

    def get_mesh(self):
        return self._mesh

    def _setup_multichip(self):
        xr.use_spmd()

    def _create_mesh(self, mesh_shape, axis_names):
        import torch_xla.distributed.spmd as xs

        device_ids = list(range(xr.global_runtime_device_count()))
        return xs.Mesh(device_ids, tuple(mesh_shape), tuple(axis_names))

    def init_process_group(
        self,
        *args,
        backend=None,
        rank=-1,
        world_size=-1,
        mesh_shape=None,
        mesh_axis_names=None,
        auto_mesh=False,
        **kwargs,
    ):
        num_devices = xr.global_runtime_device_count()
        if mesh_shape is not None:
            if mesh_axis_names is None:
                raise ValueError("mesh_axis_names is required when mesh_shape is set")
        elif auto_mesh and num_devices > 1:
            mesh_shape = (1, num_devices)
            mesh_axis_names = ("batch", "model")

        if mesh_shape is not None:
            self._setup_multichip()
            self._mesh = self._create_mesh(mesh_shape, mesh_axis_names)
            return self._mesh

        # xla:// is registered by torch_xla as a custom rendezvous URL scheme.
        dist_kwargs = dict(kwargs)
        dist_kwargs.setdefault("init_method", "xla://")
        if rank != -1:
            dist_kwargs["rank"] = rank
        if world_size != -1:
            dist_kwargs["world_size"] = world_size
        torch.distributed.init_process_group(
            *args,
            backend=backend or self.ccl,
            **dist_kwargs,
        )
        return None


impl = XlaDevice()
