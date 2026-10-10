"""Command line interface:  ``autoinsight run | serve | predict | doctor``."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _print_event(event: dict) -> None:
    icon = {"ok": "+", "retry": "~", "failed": "x", "skipped": "-"}.get(event["status"], ".")
    print(f"  [{icon}] {event['agent']:<9} {event['message']}", flush=True)


def cmd_run(args: argparse.Namespace) -> int:
    from .config import Settings
    from .pipeline import run_pipeline
    from .sandbox import SandboxUnavailable

    for flag, env in (("provider", "LLM_PROVIDER"), ("model", "MODEL"), ("sandbox", "SANDBOX_MODE"), ("runs_dir", "RUNS_DIR")):
        if getattr(args, flag):
            os.environ[f"AUTOINSIGHT_{env}"] = str(getattr(args, flag))
    settings = Settings.from_env()
    print(f"AutoInsight - provider={settings.resolved_provider()}  sandbox={settings.sandbox_mode}")
    try:
        result = run_pipeline(args.csv, args.question, target=args.target, settings=settings, on_event=_print_event)
    except SandboxUnavailable as exc:
        print(f"\nRefusing to run: {exc}", file=sys.stderr)
        return 2
    print()
    if not result.ok:
        print(f"FAILED: {result.error}", file=sys.stderr)
        print(f"Details: {result.run_dir / 'logs'}", file=sys.stderr)
        return 1
    s = result.summary
    print(f"Done in {s['duration_s']}s  ->  {result.run_dir}")
    print(f"  target: {s['target']} ({s['task_type']})   top drivers: {', '.join(s['top_drivers'])}")
    if s.get("model"):
        print("  model : " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s["model"].items()))
    print(f"  report: {result.report_path}")
    print(f"  html  : {result.run_dir / 'artifacts' / 'report.html'}")
    print(f"  predict later with:  autoinsight predict {result.run_id} --json '{{\"feature\": value}}'")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Install the API extra:  pip install 'autoinsight[api]'", file=sys.stderr)
        return 2
    if args.runs_dir:
        os.environ["AUTOINSIGHT_RUNS_DIR"] = args.runs_dir
    uvicorn.run("autoinsight.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .config import Settings
    from .predictor import PredictionError, Predictor

    settings = Settings.from_env()
    art = (Path(args.runs_dir) if args.runs_dir else settings.runs_dir) / args.run_id / "artifacts"
    try:
        predictor = Predictor.load(art)
        record = json.loads(args.json)
        results = predictor.predict(record if isinstance(record, list) else [record])
    except (FileNotFoundError, PredictionError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(results, indent=2))
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    from . import __version__
    from .config import Settings
    from .sandbox import docker_ready

    s = Settings.from_env()
    provider = s.resolved_provider()
    rows = [
        ("AutoInsight", __version__),
        ("Python", sys.version.split()[0]),
        ("LLM provider", f"{provider}" + ("" if provider != "offline" else "  (set ANTHROPIC_API_KEY or OPENAI_API_KEY to use a real LLM)")),
        ("Docker CLI", "found" if shutil.which("docker") else "NOT found"),
        ("Sandbox image", f"{s.sandbox_image}: " + ("ready" if docker_ready(s.sandbox_image) else "not available -> local sandbox (templates only)")),
        ("Runs directory", str(s.runs_dir.resolve())),
    ]
    for key, value in rows:
        print(f"{key:<15} {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autoinsight", description="Multi-agent automated data analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="analyse a CSV file")
    run.add_argument("csv", help="path to the CSV file")
    run.add_argument("-q", "--question", required=True, help='e.g. "Find the driving factors behind customer churn"')
    run.add_argument("-t", "--target", help="exact name of the target column (optional)")
    run.add_argument("--provider", choices=["auto", "anthropic", "openai", "offline"], help="LLM provider (default: auto)")
    run.add_argument("--model", help="model name override")
    run.add_argument("--sandbox", choices=["auto", "docker", "local"], help="code execution sandbox (default: auto)")
    run.add_argument("--runs-dir", help="where to store runs (default ./runs)")
    run.set_defaults(func=cmd_run)

    serve = sub.add_parser("serve", help="start the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--runs-dir")
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    predict = sub.add_parser("predict", help="score a record with the model of a finished run")
    predict.add_argument("run_id")
    predict.add_argument("--json", required=True, help="a JSON object (one record) or a list of objects")
    predict.add_argument("--runs-dir")
    predict.set_defaults(func=cmd_predict)

    sub.add_parser("doctor", help="check the environment").set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
