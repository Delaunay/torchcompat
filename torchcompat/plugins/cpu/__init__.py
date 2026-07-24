"""CPU compatibility layer."""

import torch

from torchcompat.utils.device import Event, TorchBackendDevice


class CpuDevice(TorchBackendDevice):
    def __init__(self):
        super().__init__(torch.cpu)

    @property
    def name(self) -> str:
        return "cpu"

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def ccl(self) -> str:
        return "gloo"

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def set_device(self, device) -> None:
        pass

    def synchronize(self, *args, **kwargs) -> None:
        pass

    @property
    def Event(self):
        return Event

    @property
    def amp(self):
        # Prefer float32 on CPU: autocast is opt-in (``enabled=True``) instead of
        # the accelerator default where ``autocast()`` enables low precision.
        class amp:
            @staticmethod
            def autocast(*args, enabled=False, device_type=None, **kwargs):
                import torch

                return torch.amp.autocast(
                    device_type or "cpu", *args, enabled=enabled, **kwargs
                )

            @staticmethod
            def GradScaler(*args, device=None, **kwargs):
                import torch

                return torch.amp.GradScaler(
                    *args, device=device or "cpu", **kwargs
                )

        return amp

    def compile(self, model, *args, backend=None, **kwargs):
        # Default to eager on CPU — inductor compile is slow and rarely useful here.
        return super().compile(model, *args, backend=backend or "eager", **kwargs)


impl = CpuDevice()
