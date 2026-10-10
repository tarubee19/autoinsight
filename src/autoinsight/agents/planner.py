"""Agent 0 - Planner: decides *what* to analyse (target, task type, columns to exclude).

The LLM proposes a plan; the plan is then validated against the real data (does the target exist?
is the positive label a real value?). If the LLM's plan is invalid after one retry, the heuristic
plan is used, so a flaky LLM can never derail the run.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..llm import LLMRequest, extract_json
from ..profiling import PlanError, build_heuristic_plan, load_dataframe, normalise_plan
from ..prompts import planner_prompts
from ..state import AgentState, RunContext


def planner_node(state: AgentState, config) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    df = load_dataframe(ctx.input_csv)
    profile, question, hint = state["profile"], state["question"], state.get("target_hint")

    heuristic, heuristic_error = None, None
    try:
        heuristic = build_heuristic_plan(df, profile, question, hint)
    except PlanError as exc:
        heuristic_error = str(exc)

    plan, note = None, ""
    if hint and heuristic is None:  # an explicit but unusable target is always the user's problem
        return _fail(ctx, heuristic_error)
    if ctx.llm.is_offline:
        plan = heuristic
    else:
        feedback = None
        for _ in range(2):
            system, user = planner_prompts(question, profile, list(df.columns), hint)
            if feedback:
                user += f"\n\nYour previous answer was rejected: {feedback}\nReturn corrected JSON."
            try:
                raw = ctx.llm.complete(LLMRequest(role="planner", system=system, user=user, json_mode=True, context={"heuristic_plan": heuristic}))
                plan = normalise_plan(extract_json(raw), df, profile, question, hint)
                break
            except ValueError as exc:  # includes PlanError
                feedback = str(exc)
            except Exception as exc:  # noqa: BLE001 - network / auth problems
                feedback = f"LLM call failed: {type(exc).__name__}: {exc}"
                break
        if plan is None and heuristic is not None:
            plan, note = heuristic, f" (LLM plan unusable - used heuristic plan: {feedback})"
    if plan is None:
        return _fail(ctx, heuristic_error or "Could not build an analysis plan.")

    plan["seed"] = ctx.settings.seed
    plan["target_selected_by"] = "user" if hint else plan["source"]
    summary = f"planner: target '{plan['target']}' ({plan['task_type']}), source={plan['source']}{note}"
    event = {"time": datetime.now(UTC).isoformat(timespec="seconds"), "agent": "planner", "status": "ok",
             "attempt": 1, "source": plan["source"], "message": summary}  # fmt: skip
    ctx.emit(event)
    return {"plan": plan, "agents": {"planner": {"attempts": 1, "status": "ok"}}, "events": [event]}


def _fail(ctx: RunContext, message: str) -> dict:
    event = {"time": datetime.now(UTC).isoformat(timespec="seconds"), "agent": "planner", "status": "failed",
             "attempt": 1, "source": "n/a", "message": f"planner failed: {message}"}  # fmt: skip
    ctx.emit(event)
    return {"error": message, "agents": {"planner": {"attempts": 1, "status": "failed", "last_error": message}}, "events": [event]}
