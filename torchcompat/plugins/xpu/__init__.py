"""Intel XPU support for pytorch"""

import torch

from torchcompat.core.errors import NotAvailable

if not hasattr(torch, "xpu"):
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError as err:
        raise NotAvailable("Could not import intel_extension_for_pytorch") from err


if torch.xpu.is_available():
    raise NotAvailable("torch.xpu is not available")


impl = torch.xpu

setattr(impl, "device_type", "xpu")
