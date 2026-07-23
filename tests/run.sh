#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-$(command -v python3)}"

LIBDIR="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
if [[ -n "${LIBDIR}" ]]; then
  export LD_LIBRARY_PATH="${LIBDIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-0}"

exec "${PYTHON}" -m pytest tests/ "$@"
