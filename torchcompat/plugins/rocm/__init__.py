"""ROCm compatibility layer."""

import torch

from torchcompat.utils.device import TorchBackendDevice
from torchcompat.utils.errors import NotAvailable

if not torch.cuda.is_available():
    raise NotAvailable("torch.cuda is not available")

# check that torch.cuda is in fact rocm
if not torch.version.hip:
    raise NotAvailable("torch.cuda is not rocm")


class RocmDevice(TorchBackendDevice):
    def __init__(self):
        super().__init__(torch.cuda)

    @property
    def name(self) -> str:
        return "rocm"

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def ccl(self) -> str:
        return "nccl"


impl = RocmDevice()
