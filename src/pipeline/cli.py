"""Single CLI entry point for the pipeline. Commands are added here as stages are implemented."""

import argparse
import json
import sys
from pathlib import Path

from langgraph.types import Command

from pipeline.config import configure_langsmith
from pipeline.graph.base import build_graph
from pipeline.graph.state import Run
from pipeline.storage import save_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the pipeline on a script idea")
    run_parser.add_argument("script", nargs="?", help="1-3 sentence idea, or full script text")
    run_parser.add_argument("--file", "-f", type=Path, help="Path to a file containing the script text")
    run_parser.add_argument(
        "--enhance-script",
        action="store_true",
        help="Enable the optional script enhancer stage (asks clarifying questions before proceeding)",
    )
    run_parser.add_argument(
        "--revision-limit",
        type=int,
        help="Max script enhancer rounds before proceeding with the latest version",
    )
    run_parser.add_argument(
        "--tarantino-threshold",
        type=float,
        help="Minimum Tarantino quality-gate score required to pass (0-1)",
    )
    run_parser.add_argument(
        "--tarantino-retry-limit",
        type=int,
        help="Max Tarantino evaluation rounds before shipping the best-scoring version",
    )
    run_parser.add_argument(
        "--enable-auto-rewriter",
        action="store_true",
        help="Auto-rewrite the script on a failed Tarantino gate instead of asking the user",
    )

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
        configure_langsmith()
        script_text = _read_script(args)
        parameters = {
            "enable_script_enhancer": args.enhance_script,
            "enable_auto_rewriter": args.enable_auto_rewriter,
        }
        if args.revision_limit is not None:
            parameters["revision_count_limit"] = args.revision_limit
        if args.tarantino_threshold is not None:
            parameters["tarantino_score_threshold"] = args.tarantino_threshold
        if args.tarantino_retry_limit is not None:
            parameters["tarantino_retry_count_limit"] = args.tarantino_retry_limit
        run = Run(script_text=script_text, parameters=parameters)
        graph = build_graph()
        config = {
            "run_name": "pipeline_run",
            "tags": ["pipeline"],
            "metadata": {"run_id": run.run_id},
            "configurable": {"thread_id": run.run_id},
        }
        try:
            result = graph.invoke(run, config=config)
            while "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                if "clarifying_questions" in payload:
                    print("\nThe script enhancer has some clarifying questions:")
                    answers = [input(f"{question}\n> ") for question in payload["clarifying_questions"]]
                    result = graph.invoke(Command(resume=answers), config=config)
                else:
                    print(f"\nTarantino gate score: {payload['score']}")
                    print(f"Feedback: {payload['feedback']}")
                    choice = input("Approve and proceed anyway, or revise the script? [approve/revise]\n> ").strip().lower()
                    if choice == "approve":
                        resume = {"decision": "approve"}
                    else:
                        revised_text = input("Enter the revised script text:\n> ")
                        resume = {"decision": "revise", "revised_text": revised_text}
                    result = graph.invoke(Command(resume=resume), config=config)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from None
        save_run(Run.model_validate({k: v for k, v in result.items() if k != "__interrupt__"}))
        print(json.dumps(result, indent=2, default=str))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
