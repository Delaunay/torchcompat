"""Inspect accelerator runtime logs."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from argklass.command import Command
from argklass.arguments import argument

from torchcompat.core.logs import log_root


def _tt_logger_file() -> Path:
    return log_root() / "tt" / "logger.txt"


def _tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return content[-lines:]


def _follow(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, 2)
        while True:
            line = handle.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
            else:
                time.sleep(0.25)


class Logs(Command):
    """Show or follow runtime logs for a backend."""

    name = "logs"

    @dataclass
    class Arguments:
        """Show or follow runtime logs for a backend."""

        backend: str = "tt"
        tail: int = argument(default=50, help="number of lines to print")
        follow: bool = False
        path_only: bool = False

    @staticmethod
    def execute(args):
        if args.backend != "tt":
            print(f"log inspection is not implemented for backend {args.backend!r}")
            return 1

        logger_file = _tt_logger_file()
        if args.path_only:
            print(f"log root:      {log_root()}")
            print(f"tt root:       {logger_file.parent}")
            print(f"logger file:   {logger_file}")
            return 0

        print(f"logger file: {logger_file}")
        print()

        if args.follow:
            print(f"--- following {logger_file} ---")
            _follow(logger_file)
            return 0

        lines = _tail(logger_file, args.tail)
        if not lines:
            print("(no log output yet)")
            return 0

        print(f"--- last {len(lines)} lines ---")
        print("\n".join(lines))
        return 0


COMMANDS = Logs
