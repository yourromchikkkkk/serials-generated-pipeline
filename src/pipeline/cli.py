"""Single CLI entry point for the pipeline. Commands are added here as stages are implemented."""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("placeholder", help="No pipeline stages are implemented yet")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "placeholder":
        print("No pipeline stages are implemented yet.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
