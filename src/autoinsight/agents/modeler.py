"""Agent 3 - Modeler: trains and validates the predictive model with the vetted trainer.

The training program is *not* LLM-written (modelling mistakes such as leakage or tuning on the
test set are too costly to leave to free-form generation). It still runs in the sandbox, and this
step is allowed to fail softly: the report is produced without a model and says why.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from .. import templates
from ..state import AgentState, RunContext
from .code_agent import _log_attempt, _promote


def modeler_node(state: AgentState, config) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    agents = {k: dict(v) for k, v in state.get("agents", {}).items()}
    code = templates.source(templates.TRAINER)
    result = ctx.sandbox.run(
        code,
        input_csv=ctx.artifacts / "cleaned.csv",
        params=ctx.sandbox_params(state["plan"]),
        exec_dir=ctx.work_dir("modeler", 1),
        timeout_s=max(ctx.settings.sandbox_timeout_s, 300),
    )
    ok = result.ok and (result.out_dir / "model.joblib").is_file() and (result.out_dir / "metrics.json").is_file()
    _log_attempt(ctx, "modeler", 1, {
        "agent": "modeler", "attempt": 1, "source": "trusted-script", "ok": ok, "error": None if ok else result.error_text(),
        "duration_s": round(result.duration_s, 2), "stdout": result.stdout, "files": result.files, "code": None,
    })  # fmt: skip
    now = datetime.now(UTC).isoformat(timespec="seconds")
    if ok:
        _promote(result.out_dir, ctx.artifacts, "modeler")
        agents["modeler"] = {"attempts": 1, "status": "ok", "last_error": None, "used_fallback": False}
        metrics = json.loads((ctx.artifacts / "metrics.json").read_text())
        headline = f"AUC {metrics['roc_auc']:.3f}" if "roc_auc" in metrics else f"R2 {metrics['r2']:.3f}"
        event = {"time": now, "agent": "modeler", "status": "ok", "attempt": 1, "source": "trusted-script",
                 "message": f"modeler trained {metrics['model']} ({headline}, {result.duration_s:.1f}s)"}  # fmt: skip
        update = {"agents": agents}
    else:
        reason = result.error_text().strip().splitlines()[-1][:300] if result.error_text().strip() else "unknown error"
        agents["modeler"] = {"attempts": 1, "status": "skipped", "last_error": reason, "used_fallback": False}
        event = {"time": now, "agent": "modeler", "status": "skipped", "attempt": 1, "source": "trusted-script",
                 "message": f"modeler skipped (no model in report): {reason}"}  # fmt: skip
        update = {"agents": agents, "model_error": reason}
    ctx.emit(event)
    update["events"] = [event]
    return update
