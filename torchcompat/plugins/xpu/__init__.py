"""Intel XPU support for pytorch"""

import torch

from torchcompat.core.errors import NotAvailable

if not hasattr(torch, "xpu"):
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError as err:
        raise NotAvailable("Could not import intel_extension_for_pytorch") from err


if not torch.xpu.is_available():
    raise NotAvailable("torch.xpu is not available")


impl = torch.xpu


def set_enable_tf32(enable=True):
    if enable:
        ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.TF32)
    else:
        ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.FP32)


# https://github.com/intel/torch-ccl?tab=readme-ov-file#usage
ccl = "ccl"


#
# XPU does NOT implement amp.GradScaler
#
class NoScale:
    def __init__(self, enabled=True) -> None:
        pass

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


if not hasattr(impl.amp, "GradScaler"):
    setattr(impl.amp, "GradScaler", NoScale)

setattr(impl, "device_type", "xpu")
setattr(impl, "set_enable_tf32", set_enable_tf32)
setattr(impl, "ccl", ccl)
