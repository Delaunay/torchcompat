"""Sysfs-based Tenstorrent hardware detection (no PJRT / torch_xla)."""

from __future__ import annotations

from pathlib import Path

SYSFS_TT_CLASS = Path("/sys/class/tenstorrent")


def _read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return default


def list_sysfs_devices() -> list[dict]:
    """Enumerate Tenstorrent devices from ``/sys/class/tenstorrent``."""
    if not SYSFS_TT_CLASS.is_dir():
        return []

    devices: list[dict] = []
    for entry in sorted(SYSFS_TT_CLASS.glob("tenstorrent!*")):
        name = entry.name
        if not name.startswith("tenstorrent!"):
            continue
        device_id = int(name.split("!", 1)[1])
        tt_path = entry.resolve()
        pci_device = tt_path.parent.parent
        bus_id = pci_device.name if pci_device.name.startswith("0000:") else ""

        devices.append(
            {
                "device_id": device_id,
                "name": name,
                "card_type": _read_text(tt_path / "tt_card_type"),
                "serial": _read_text(tt_path / "tt_serial"),
                "bus_id": bus_id,
                "sysfs_path": str(tt_path),
            }
        )

    return devices


def format_sysfs_summary(devices: list[dict]) -> str:
    """Compact human-readable summary of sysfs-visible TT devices."""
    if not devices:
        return "no Tenstorrent devices in /sys/class/tenstorrent"

    parts = []
    for device in devices:
        label = device.get("card_type") or "tenstorrent"
        parts.append(f"{device['name']} ({label}, pci={device.get('bus_id') or '?'})")
    return f"{len(devices)} device(s): " + ", ".join(parts)


def validate_visible_devices(
    visible: str | None,
    devices: list[dict] | None = None,
) -> str | None:
    """Return an error when ``TT_VISIBLE_DEVICES`` references missing chips."""
    if not visible:
        return None

    available = {device["device_id"] for device in (devices or list_sysfs_devices())}
    requested: set[int] = set()
    for part in visible.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            requested.add(int(part))
        except ValueError:
            return f"TT_VISIBLE_DEVICES contains invalid entry {part!r}"

    missing = sorted(requested - available)
    if missing:
        return (
            f"TT_VISIBLE_DEVICES requests chip(s) {missing} but sysfs shows "
            f"{sorted(available)}"
        )
    return None


def hardware_unavailable_message() -> str:
    """Explain missing sysfs-visible Tenstorrent hardware."""
    if not SYSFS_TT_CLASS.is_dir():
        return (
            "No /sys/class/tenstorrent directory. "
            "Install/load tt-kmd and confirm the PCI device is bound to the "
            "tenstorrent driver."
        )
    return (
        "No tenstorrent!* entries under /sys/class/tenstorrent. "
        "Hardware may be missing or the driver is not bound."
    )


def _read_proc_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(":")
            if not raw:
                continue
            number = raw.strip().split()[0]
            values[key.strip()] = int(number)
    except OSError:
        pass
    return values


def hugepage_status() -> dict:
    """Return hugepage availability from /proc/meminfo (kB fields as ints)."""
    meminfo = _read_proc_meminfo()
    return {
        "total": meminfo.get("HugePages_Total", 0),
        "free": meminfo.get("HugePages_Free", 0),
        "surplus": meminfo.get("HugePages_Surp", 0),
    }


def format_init_error(err: BaseException) -> str:
    """Turn common TT init failures into actionable guidance."""
    message = str(err)
    lower = message.lower()

    if "failed to pin pages for hugepage" in lower or "cannot allocate memory" in lower:
        pages = hugepage_status()
        return (
            "Tenstorrent could not allocate hugepages for host/device memory. "
            f"System hugepages: total={pages['total']}, free={pages['free']}. "
            "Wormhole/Grayskull usually need 1G hugepages configured "
            "(see tt-metal INSTALLING.md, step 3). "
            "After fixing hugepages, start a fresh Python process."
        )

    if "initializecomputationclient() can only be called once" in lower:
        return (
            "The Tenstorrent PJRT client was already initialized in this process "
            "(often after a previous failed init). Restart Python and try again."
        )

    if "chip_in_use" in lower or "waiting for lock" in lower:
        return (
            "Another process is using the Tenstorrent device. "
            "Stop other TT jobs or set TT_VISIBLE_DEVICES to a free chip, "
            "then start a fresh Python process."
        )

    return message
