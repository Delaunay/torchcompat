"""CUDA compatibility layer."""

import torch

from torchcompat.utils.device import TorchBackendDevice
from torchcompat.utils.errors import NotAvailable

if not torch.cuda.is_available():
    raise NotAvailable("torch.cuda is not available")

# check that torch.cuda is in fact cuda and NOT rocm
if not torch.version.cuda:
    raise NotAvailable("torch.cuda is not rocm")


class CudaDevice(TorchBackendDevice):
    def __init__(self):
        super().__init__(torch.cuda)

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def ccl(self) -> str:
        return "nccl"

    def set_enable_tf32(self, enable: bool = True) -> None:
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable


impl = CudaDevice()
