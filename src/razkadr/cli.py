"""Command line entry point. Subcommands are added by naryady of wave 1+."""

import argparse
import sys

from razkadr import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="razkadr", description="Video in, deck out.")
    parser.add_argument("--version", action="version", version=f"razkadr {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
