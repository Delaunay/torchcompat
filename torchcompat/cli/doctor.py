"""Check Tenstorrent runtime prerequisites."""

from __future__ import annotations

import os
from dataclasses import dataclass

from argklass.command import Command

from torchcompat.plugins.tt.sysfs import (
    format_sysfs_summary,
    hugepage_status,
    list_sysfs_devices,
    validate_visible_devices,
)


class Doctor(Command):
    """Print environment checks for accelerator backends."""

    name = "doctor"

    @dataclass
    class Arguments:
        """Print environment checks for accelerator backends."""

        backend: str = "tt"

    @staticmethod
    def execute(args):
        if args.backend != "tt":
            print(f"doctor checks are not implemented for backend {args.backend!r}")
            return 1

        pages = hugepage_status()
        sysfs_devices = list_sysfs_devices()
        visible_error = validate_visible_devices(
            os.environ.get("TT_VISIBLE_DEVICES"),
            sysfs_devices,
        )

        print("Tenstorrent checks:")
        if sysfs_devices:
            print(f"  Sysfs:           {format_sysfs_summary(sysfs_devices)}")
        else:
            print("  Sysfs:           no devices under /sys/class/tenstorrent")
        print(f"  HugePages_Total: {pages['total']}")
        print(f"  HugePages_Free:  {pages['free']}")

        ok = True
        if not sysfs_devices:
            ok = False
            print()
            print("  FAIL: no sysfs-visible Tenstorrent devices.")
            print("  Check tt-kmd is loaded and PCI devices are bound.")
        if visible_error:
            ok = False
            print()
            print(f"  FAIL: {visible_error}")
        if pages["total"] == 0:
            ok = False
            print()
            print("  FAIL: no hugepages reserved.")
            print("  Wormhole/Grayskull need 1G hugepages for host/device memory.")
            print("  See tt-metal INSTALLING.md step 3, then restart Python.")
        else:
            print("  OK: hugepages are configured.")

        print()
        print("If TT init failed once in this shell, restart Python before retrying.")
        print("PJRT/XLA can only initialize once per process.")
        return 0 if ok else 1


COMMANDS = Doctor
