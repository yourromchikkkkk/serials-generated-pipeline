"""Single CLI entry point for the pipeline. Commands are added here as stages are implemented."""

import argparse
import json
import sys
from pathlib import Path

from pipeline.graph.base import build_graph
from pipeline.graph.state import Run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the pipeline on a script idea")
    run_parser.add_argument("script", nargs="?", help="1-3 sentence idea, or full script text")
    run_parser.add_argument("--file", "-f", type=Path, help="Path to a file containing the script text")

    return parser


def _read_script(args: argparse.Namespace) -> str:
    if args.script and args.file:
        raise SystemExit("Provide either a script argument or --file, not both")
    if args.file:
        if not args.file.exists():
            raise SystemExit(f"File not found: {args.file}")
        return args.file.read_text()
    if args.script:
        return args.script
    raise SystemExit("Provide a script argument or --file")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        script_text = _read_script(args)
        graph = build_graph()
        try:
            result = graph.invoke(Run(script_text=script_text))
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from None
        print(json.dumps(result, indent=2, default=str))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
