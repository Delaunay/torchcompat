"""List available torchcompat backends."""

from __future__ import annotations

import json
from dataclasses import dataclass

from argklass.command import Command

from torchcompat.core.load import backend_status


class Backends(Command):
    """Discover which accelerator plugins are available on this machine."""

    name = "backends"

    @dataclass
    class Arguments:
        """Discover which accelerator plugins are available on this machine."""

        json: bool = False

    @staticmethod
    def execute(args):
        results = backend_status()

        if args.json:
            print(json.dumps(results, indent=2))
            return 0

        available = []
        unavailable = []
        for name in sorted(results):
            probe = results[name]
            if probe.get("ok"):
                device = probe.get("device", probe.get("device_type", "?"))
                available.append(f"  {name:8}  {device}")
            else:
                error = probe.get("error", "unavailable")
                unavailable.append(f"  {name:8}  {error}")

        if available:
            print("Available:")
            print("\n".join(available))
        if unavailable:
            if available:
                print()
            print("Unavailable:")
            print("\n".join(unavailable))

        return 0


COMMANDS = Backends
