"""Generic 'write code -> guard -> sandbox -> validate -> repair' step shared by the Data Engineer
and the Analyst.

One call of :func:`run_code_agent_step` performs ONE attempt. The LangGraph edge after the node
decides what happens next: success -> next agent, retry -> same node again (a graph cycle), failed
-> failure node. Attempt budget: ``max_code_attempts`` LLM attempts, then one attempt with the
vetted template (graceful degradation), then give up.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .. import templates
from ..code_guard import check_code
from ..llm import LLMRequest, extract_code
from ..state import AgentState, RunContext


@dataclass
class CodeAgentSpec:
    name: str  # engineer | analyst  (also the LLMRequest role)
    template: str  # vetted fallback program
    build_prompt: Callable[[AgentState, str | None, str | None], tuple[str, str]]
    input_csv: Callable[[RunContext], Path]
    validate: Callable[[Path, AgentState, RunContext], list[str]]
    post_process: Callable[[Path], None] | None = None
    timeout_s: int | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log_attempt(ctx: RunContext, name: str, attempt: int, payload: dict) -> None:
    (ctx.logs / f"{name}_attempt{attempt}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if payload.get("code"):
        (ctx.logs / f"{name}_attempt{attempt}.py").write_text(payload["code"], encoding="utf-8")


# Files each step owns. Nothing else may write these names into ``artifacts/`` - e.g. the Analyst must not be
# able to replace the Data Engineer's cleaned.csv, and no code-writing agent may supply model.joblib (which the
# API later unpickles).
OWNED_FILES = {
    "engineer": {"cleaned.csv", "cleaning_log.json"},
    "analyst": {"stats.json"},
    "modeler": {"model.joblib", "metrics.json", "feature_importance.json", "schema.json"},
}
RESERVED_FILES = {"report.md", "report.html", "facts.json", "grounding.json"}.union(*OWNED_FILES.values())
PLAIN_SUFFIXES = {".csv", ".json", ".png"}


def _promote(out_dir: Path, artifacts: Path, agent: str) -> list[str]:
    """Copy a sandbox output directory into ``artifacts/``, accepting only what ``agent`` may write.

    Allowed: regular files with a data suffix (.csv/.json/.png; .joblib only for the modeler), not shadowing a
    file owned by another step. Returns the relative paths that were skipped (they stay in ``work/`` for audit).
    """
    owned = OWNED_FILES.get(agent, set())
    allowed_suffixes = PLAIN_SUFFIXES | ({".joblib"} if agent == "modeler" else set())
    skipped: list[str] = []
    for src in sorted(out_dir.rglob("*")):
        rel = src.relative_to(out_dir)
        if src.is_dir() or ".mpl" in rel.parts or "__pycache__" in rel.parts:
            continue
        name = rel.as_posix()
        protected = name in RESERVED_FILES and name not in owned
        if src.is_symlink() or not src.is_file() or src.suffix.lower() not in allowed_suffixes or protected or len(rel.parts) > 2:
            skipped.append(name)
            continue
        dest = artifacts / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    return skipped


def run_code_agent_step(ctx: RunContext, state: AgentState, spec: CodeAgentSpec) -> dict:
    agents = {k: dict(v) for k, v in state.get("agents", {}).items()}
    st = agents.setdefault(spec.name, {"attempts": 0, "status": "pending"})
    attempt = int(st.get("attempts", 0)) + 1
    on_fallback = attempt > ctx.settings.max_code_attempts and not ctx.llm.is_offline
    source = "template" if on_fallback else ("offline-template" if ctx.llm.is_offline else "llm")
    started = time.monotonic()
    code: str | None = None
    error: str | None = None
    result = None
    guard_msg = None

    # 1. get code ------------------------------------------------------------------------------
    if on_fallback:
        code = templates.source(spec.template)
    else:
        system, user = spec.build_prompt(state, st.get("last_code"), st.get("last_error"))
        request = LLMRequest(role=spec.name, system=system, user=user, context={"attempt": attempt})
        try:
            code = extract_code(ctx.llm.complete(request))
        except Exception as exc:  # noqa: BLE001 - network/auth/rate-limit errors etc.
            error = f"LLM call failed: {type(exc).__name__}: {exc}"

    # 2. static safety check --------------------------------------------------------------------
    if code and error is None:
        guard = check_code(code)
        if not guard.ok:
            guard_msg = guard.message()
            error = "Your program was rejected by the safety guard and NOT executed: " + guard_msg

    # 3. run in the sandbox -----------------------------------------------------------------------
    if code and error is None:
        result = ctx.sandbox.run(
            code,
            input_csv=spec.input_csv(ctx),
            params=ctx.sandbox_params(state["plan"]),
            exec_dir=ctx.work_dir(spec.name, attempt),
            timeout_s=spec.timeout_s,
        )
        if not result.ok:
            error = result.error_text()
        else:
            # 4. contract validation -----------------------------------------------------------------
            problems = spec.validate(result.out_dir, state, ctx)
            if problems:
                error = "The program ran but its output breaks the contract:\n- " + "\n- ".join(problems)

    duration = time.monotonic() - started
    _log_attempt(
        ctx, spec.name, attempt,
        {
            "agent": spec.name, "attempt": attempt, "source": source, "time": _now(),
            "duration_s": round(duration, 2), "ok": error is None, "error": error, "guard": guard_msg,
            "sandbox_mode": ctx.sandbox.mode, "stdout": getattr(result, "stdout", ""),
            "files": getattr(result, "files", []), "code": code,
        },
    )  # fmt: skip

    # 5. decide ------------------------------------------------------------------------------------
    if error is None:
        assert result is not None
        if spec.post_process:
            spec.post_process(result.out_dir)
        skipped = _promote(result.out_dir, ctx.artifacts, spec.name)
        st.update(status="ok", attempts=attempt, last_error=None, last_code=code, used_fallback=on_fallback)
        message = f"{spec.name} finished (attempt {attempt}, {source}, {duration:.1f}s)"
        if skipped:
            message += f"; ignored {len(skipped)} disallowed output file(s): {', '.join(skipped[:3])}"
        update: dict = {"agents": agents}
        status = "ok"
    else:
        final = on_fallback or ctx.llm.is_offline
        st.update(status="failed" if final else "retry", attempts=attempt, last_error=error, last_code=code or st.get("last_code"))
        message = f"{spec.name} attempt {attempt} ({source}) failed: {error.splitlines()[0][:200]}"
        update = {"agents": agents}
        status = st["status"]
        if final:
            update["error"] = f"The {spec.name} agent could not produce valid output after {attempt} attempts. Last error: {error[:600]}"

    event = {"time": _now(), "agent": spec.name, "status": status, "attempt": attempt, "source": source, "message": message}
    ctx.emit(event)
    update["events"] = [event]
    return update
