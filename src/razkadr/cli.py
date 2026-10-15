"""Command line entry point. Subcommands are added wave by wave."""

import argparse
import json
import sys
from pathlib import Path

from razkadr import __version__

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_FILE_ERROR = 2


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a Storyboard JSON file. 0 = ok, 1 = invalid, 2 = file error."""
    path = Path(args.file)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"razkadr validate: cannot read {path}: {exc.strerror or exc}", file=sys.stderr)
        return EXIT_FILE_ERROR
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"razkadr validate: invalid JSON in {path}: {exc}", file=sys.stderr)
        return EXIT_FILE_ERROR

    from razkadr.validation import validate_storyboard

    issues = validate_storyboard(data)
    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        return EXIT_INVALID
    print("ok")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="razkadr", description="Video in, deck out.")
    parser.add_argument("--version", action="version", version=f"razkadr {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_validate = sub.add_parser("validate", help="Validate a Storyboard JSON file")
    p_validate.add_argument("file", help="Path to the Storyboard JSON file")
    p_validate.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_OK
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
