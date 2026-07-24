"""Plugin template — subclass ``Device`` and export ``impl``."""

from torchcompat.utils.device import Device
from torchcompat.utils.errors import NotAvailable

raise NotAvailable("template plugin is not a real backend")


class TemplateDevice(Device):
    @property
    def name(self) -> str:
        return "template"

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def ccl(self) -> str:
        return "gloo"


# Unreachable — kept as documentation for new plugins:
# impl = TemplateDevice()
