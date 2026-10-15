"""Command line entry point.

Subcommands: validate (real, naryad 002/035) and stubs for the pipeline
commands (index/select/text/build/eval/eval-check) with a unified
interface --config/--force/--runs-dir/--resume; their logic arrives with
later naryads. Exit codes: 0 ok, 1 invalid document, 2 file/IO error,
3 command not implemented.
"""

import argparse
import json
import sys
from pathlib import Path

from razkadr import __version__

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_FILE_ERROR = 2
EXIT_NOT_IMPLEMENTED = 3

# stub command -> owning naryad (shown in help and stub message)
STUB_COMMANDS = {
    "index": "011",
    "select": "014",
    "text": "017",
    "build": "028",
    "eval": "015",
    "eval-check": "003",
}


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a JSON document. 0 = ok, 1 = invalid, 2 = file error.

    Without --kind the file is validated as a Storyboard (naryad 002);
    --kind selects a stage artifact schema (naryad 035, docs/artifacts.md).
    """
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

    from razkadr.validation import validate_document

    issues = validate_document(data, kind=args.kind)
    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        return EXIT_INVALID
    print("ok")
    return EXIT_OK


def _cmd_stub(args: argparse.Namespace) -> int:
    print(
        f"razkadr {args.cmd}: not implemented yet (planned: naryad No{args.naryad})",
        file=sys.stderr,
    )
    return EXIT_NOT_IMPLEMENTED


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Path to razkadr.toml (default: search)")
    parser.add_argument("--force", action="store_true", help="Re-run stages, ignore stage cache")
    parser.add_argument("--runs-dir", default=None, help="Directory for runs (default: config runs_dir)")
    parser.add_argument("--resume", default=None, metavar="RUN_ID", help="Reuse an existing run directory")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="razkadr", description="Video in, deck out.")
    parser.add_argument("--version", action="version", version=f"razkadr {__version__}")
    parser.add_argument("--verbose", action="store_true", help="Debug logging (JSON lines to stderr)")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    from razkadr.validation import KINDS

    p_validate = sub.add_parser("validate", help="Validate a Storyboard or artifact JSON file")
    p_validate.add_argument(
        "--kind",
        choices=sorted(KINDS),
        default="storyboard",
        help="Document kind: storyboard (default) or a stage artifact",
    )
    p_validate.add_argument("file", help="Path to the JSON file")
    p_validate.set_defaults(func=_cmd_validate)

    for name, naryad in STUB_COMMANDS.items():
        p_stub = sub.add_parser(name, help=f"[stub] planned by naryad No{naryad}")
        _add_common_args(p_stub)
        p_stub.set_defaults(func=_cmd_stub, cmd=name, naryad=naryad)

    args = parser.parse_args(argv)

    from razkadr.log import setup_logging

    setup_logging(verbose=getattr(args, "verbose", False))

    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_OK
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
