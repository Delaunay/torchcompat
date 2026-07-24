"""Re-export TT sysfs helpers (prefer ``torchcompat.utils.tt_sysfs``).

Importing this module still runs ``torchcompat.plugins.tt`` package init, which
raises ``NotAvailable`` when no Tenstorrent hardware is present. Callers that
must work on non-TT hosts (CLI, XLA deferral, log setup) should import from
``torchcompat.utils.tt_sysfs`` instead.
"""

from torchcompat.utils.tt_sysfs import *  # noqa: F403
from torchcompat.utils.tt_sysfs import (  # noqa: F401
    SYSFS_TT_CLASS,
    format_init_error,
    format_sysfs_summary,
    hardware_unavailable_message,
    hugepage_status,
    list_sysfs_devices,
    validate_visible_devices,
)
