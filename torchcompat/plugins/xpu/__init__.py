"""Intel XPU support for pytorch."""

import torch

from torchcompat.utils.device import TorchBackendDevice
from torchcompat.utils.errors import NotAvailable

ipex = None
if not hasattr(torch, "xpu"):
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError as err:
        raise NotAvailable("Could not import intel_extension_for_pytorch") from err


if not torch.xpu.is_available():
    raise NotAvailable("torch.xpu is not available")


class NoScale:
    def __init__(self, enabled=True) -> None:
        pass

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


class XpuDevice(TorchBackendDevice):
    def __init__(self):
        super().__init__(torch.xpu)
        if not hasattr(self._backend.amp, "GradScaler"):
            self._backend.amp.GradScaler = NoScale

    @property
    def name(self) -> str:
        return "xpu"

    @property
    def device_type(self) -> str:
        return "xpu"

    @property
    def ccl(self) -> str:
        # https://github.com/intel/torch-ccl?tab=readme-ov-file#usage
        return "ccl"

    def set_enable_tf32(self, enable: bool = True) -> None:
        if ipex is None:
            return
        if enable:
            ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.TF32)
        else:
            ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.FP32)


impl = XpuDevice()
