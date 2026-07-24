"""torchcompat CLI entry point using argklass for command discovery."""

from __future__ import annotations

import argparse
import sys

from argklass.argformat import HelpAction, HelpActionException
from argklass.command import ParentCommand
from argklass.plugin import discover_module_commands_no_cache


def discover_commands():
    import torchcompat.cli

    return discover_module_commands_no_cache(torchcompat.cli, None).found_commands


def build_parser(commands):
    parser = argparse.ArgumentParser(
        prog="torchcompat",
        add_help=False,
        description="Discover accelerator backends and inspect runtime logs",
    )
    parser.add_argument(
        "-h", "--help", action=HelpAction, help="show this help message and exit"
    )

    subparsers = parser.add_subparsers(dest="command")
    ParentCommand.dispatch = {}
    for command in commands.values():
        command.arguments(subparsers)

    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv = [str(x) for x in argv]

    commands = discover_commands()

    try:
        parser = build_parser(commands)
        parsed_args = parser.parse_args(argv)
    except HelpActionException:
        return 0

    cmd_name = parsed_args.command
    if cmd_name is None:
        parser.print_usage()
        return 1

    command = commands.get(cmd_name)
    if command is None:
        print(f"Action `{cmd_name}` not implemented")
        return 1

    try:
        returncode = command.execute(parsed_args)
        return returncode if returncode is not None else 0
    except KeyboardInterrupt:
        return 130
