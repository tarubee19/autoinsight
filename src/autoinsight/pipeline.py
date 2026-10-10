"""High-level entry point: ``run_pipeline(csv, question)`` -> a finished run directory.

Run directory layout::

    runs/<run_id>/
        input.csv                original upload (copy)
        state.json               live progress (polled by the API/UI)
        summary.json             machine-readable outcome
        artifacts/               report.md, report.html, cleaned.csv, stats.json, metrics.json,
                                 model.joblib, schema.json, charts/*.png, facts.json, ...
        logs/                    every code attempt of every agent (code, stdout, error)
        work/                    raw sandbox working directories (safe to delete)
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .graph import build_graph
from .llm import LLMClient, build_llm
from .profiling import PlanError, load_dataframe
from .sandbox import Sandbox
from .state import STAGES, RunContext

NODE_TO_STAGE = {"profile": "profile", "planner": "planner", "engineer": "engineer", "analyst": "analyst",
                 "modeler": "modeler", "reporter": "reporter"}  # fmt: skip
NEXT_STAGE = {s: STAGES[i + 1] if i + 1 < len(STAGES) else None for i, s in enumerate(STAGES)}


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    status: str  # completed | failed
    error: str | None
    summary: dict

    @property
    def report_path(self) -> Path:
        return self.run_dir / "artifacts" / "report.md"

    @property
    def ok(self) -> bool:
        return self.status == "completed"


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def write_state(run_dir: Path, payload: dict) -> None:
    tmp = run_dir / "state.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(run_dir / "state.json")  # atomic: pollers never see half a file


def run_pipeline(
    csv_path: str | Path,
    question: str,
    *,
    target: str | None = None,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    sandbox: Sandbox | None = None,
    run_id: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    keep_work_dir: bool = False,
    state_extra: dict | None = None,
) -> RunResult:
    settings = settings or Settings.from_env()
    llm = llm or build_llm(settings)
    sandbox = sandbox or Sandbox(settings)
    sandbox.assert_safe_for(llm.is_offline)  # refuse LLM code in a weak sandbox

    run_id = run_id or new_run_id()
    run_dir = (settings.runs_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    # Canonicalise the upload once (comma-separated UTF-8, stripped headers) so every downstream
    # reader - vetted templates and LLM-written code alike - sees a standard CSV. Values are untouched.
    try:
        load_dataframe(csv_path, normalise_missing=False).to_csv(run_dir / "input.csv", index=False)
    except PlanError as exc:  # unreadable / empty / duplicate headers: fail fast with a clear message
        error = str(exc)
        write_state(run_dir, {"run_id": run_id, "status": "failed", "error": error, "question": question,
                              "stages": {s: "pending" for s in STAGES}, "events": [], **(state_extra or {})})  # fmt: skip
        (run_dir / "summary.json").write_text(json.dumps({"status": "failed", "error": error}, indent=2), encoding="utf-8")
        return RunResult(run_id, run_dir, "failed", error, {"status": "failed", "error": error})

    events: list[dict] = []
    stages = {s: "pending" for s in STAGES}
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    def snapshot(status: str, error: str | None = None) -> None:
        write_state(run_dir, {
            "run_id": run_id, "status": status, "error": error, "question": question, "stages": stages,
            "started_at": started_at, "elapsed_s": round(time.monotonic() - started, 1),
            "llm": llm.name, "sandbox": sandbox.mode, "events": events, **(state_extra or {}),
        })  # fmt: skip

    def emit(event: dict) -> None:
        events.append(event)
        if on_event:
            on_event(event)

    ctx = RunContext(settings=settings, llm=llm, sandbox=sandbox, run_dir=run_dir, emit=emit)
    graph = build_graph()
    initial = {"question": question, "target_hint": target, "agents": {}, "events": []}
    stages["profile"] = "running"
    snapshot("running")

    final: dict = {}
    try:
        for step in graph.stream(initial, {"configurable": {"ctx": ctx}, "recursion_limit": 80}, stream_mode="updates"):
            for node, update in step.items():
                final.update({k: v for k, v in (update or {}).items() if k != "events"})
                stage = NODE_TO_STAGE.get(node)
                if stage:
                    agent_status = (update or {}).get("agents", {}).get(stage, {}).get("status") if update else None
                    if update and update.get("error"):
                        stages[stage] = "failed"
                    elif agent_status == "retry":
                        stages[stage] = "running"  # self-loop: same stage runs again
                    else:
                        stages[stage] = "skipped" if agent_status == "skipped" else "done"
                        nxt = NEXT_STAGE.get(stage)
                        if nxt and stages[nxt] == "pending":
                            stages[nxt] = "running"
            snapshot("running")
    except Exception as exc:  # noqa: BLE001 - never leave a run stuck in "running"
        error = f"Unexpected internal error: {type(exc).__name__}: {exc}"
        (run_dir / "summary.json").write_text(json.dumps({"status": "failed", "error": error}, indent=2), encoding="utf-8")
        snapshot("failed", error)
        return RunResult(run_id, run_dir, "failed", error, {"status": "failed", "error": error})

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["duration_s"] = round(time.monotonic() - started, 1)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    status = summary["status"]
    snapshot(status, summary.get("error"))
    if not keep_work_dir:
        shutil.rmtree(run_dir / "work", ignore_errors=True)
    return RunResult(run_id, run_dir, status, summary.get("error"), summary)
