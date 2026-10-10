"""Bookkeeping nodes: profile (entry), finalize (success) and fail (error exit)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from ..profiling import PlanError, load_dataframe, profile_dataframe
from ..state import AgentState, RunContext


def _event(agent: str, status: str, message: str) -> dict:
    return {"time": datetime.now(UTC).isoformat(timespec="seconds"), "agent": agent, "status": status,
            "attempt": 1, "source": "system", "message": message}  # fmt: skip


def profile_node(state: AgentState, config) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    try:
        df = load_dataframe(ctx.input_csv)
    except PlanError as exc:
        event = _event("profile", "failed", f"profile failed: {exc}")
        ctx.emit(event)
        return {"error": str(exc), "events": [event]}
    profile = profile_dataframe(df, send_samples=ctx.settings.send_sample_values)
    event = _event("profile", "ok", f"profiled {profile['n_rows']:,} rows x {profile['n_cols']} columns")
    ctx.emit(event)
    return {"profile": profile, "agents": {}, "events": [event]}


def finalize_node(state: AgentState, config) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    art = ctx.artifacts
    metrics = json.loads((art / "metrics.json").read_text()) if (art / "metrics.json").is_file() else None
    stats = json.loads((art / "stats.json").read_text())
    headline = None
    if metrics:
        headline = {k: metrics[k] for k in ("model", "roc_auc", "pr_auc", "recall", "precision", "r2", "mae") if k in metrics}
    summary = {
        "status": "completed",
        "question": state["question"],
        "target": state["plan"]["target"],
        "task_type": state["plan"]["task_type"],
        "plan_source": state["plan"].get("source"),
        "top_drivers": [d["feature"] for d in stats.get("top_drivers", [])[:5]],
        "model": headline,
        "model_available": (art / "model.joblib").is_file(),
        "grounding": state.get("grounding"),
        "llm": ctx.llm.name,
        "sandbox": ctx.sandbox.mode,
        "attempts": {k: v.get("attempts", 0) for k, v in state.get("agents", {}).items()},
        "fallbacks": [k for k, v in state.get("agents", {}).items() if v.get("used_fallback")],
        "artifacts": sorted(str(p.relative_to(art)) for p in art.rglob("*") if p.is_file()),
    }
    (ctx.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    event = _event("finalize", "ok", "run completed")
    ctx.emit(event)
    return {"outcome": "completed", "events": [event]}


def fail_node(state: AgentState, config) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    message = state.get("error") or "Unknown failure"
    last = {k: v.get("last_error") for k, v in state.get("agents", {}).items() if v.get("last_error")}
    report = (
        "# Analysis could not be completed\n\n"
        f"**Question:** {state.get('question')}\n\n**Reason:** {message}\n\n"
        "Check `logs/` for the code and error of every attempt.\n"
    )
    (ctx.artifacts / "report.md").write_text(report, encoding="utf-8")
    summary = {"status": "failed", "question": state.get("question"), "error": message, "agent_errors": last, "llm": ctx.llm.name}
    (ctx.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    event = _event("fail", "failed", f"run failed: {message[:300]}")
    ctx.emit(event)
    return {"outcome": "failed", "events": [event]}
