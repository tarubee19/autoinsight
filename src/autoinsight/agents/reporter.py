"""Agent 4 - Reporter: turns the computed results into a business report.

Anti-hallucination design:
  1. The reporter only sees a compact ``facts`` JSON, never raw data.
  2. Every number in its narrative is traced back to those facts (``grounding.check_numbers``).
  3. If numbers cannot be traced, it gets one chance to fix them; after that the deterministic
     narrative writer is used. The final report states the verified-number rate.
"""

from __future__ import annotations

import json
import platform
import re
from datetime import UTC, datetime

from .. import offline
from ..grounding import check_numbers
from ..llm import LLMRequest
from ..prompts import reporter_prompts
from ..reporting import assemble_report, build_facts, to_html, utc_now
from ..state import AgentState, RunContext

REQUIRED_SECTIONS = [
    "Executive summary", "Data quality", "What drives", "Predictive model", "Recommendations", "Limitations",
]  # fmt: skip
MIN_GROUNDING = 0.9


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sections_ok(text: str) -> list[str]:
    headings = re.findall(r"^##\s+(.*)$", text, re.MULTILINE)
    return [s for s in REQUIRED_SECTIONS if not any(h.lower().startswith(s.lower()) for h in headings)]


def reporter_node(state: AgentState, config) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    art = ctx.artifacts
    cleaning, stats = _read(art / "cleaning_log.json"), _read(art / "stats.json")
    metrics, importance = _read(art / "metrics.json"), _read(art / "feature_importance.json")
    facts = build_facts(
        question=state["question"], plan=state["plan"], cleaning=cleaning, stats=stats,
        metrics=metrics, importance=importance, model_error=state.get("model_error"),
    )  # fmt: skip

    narrative, grounding, used_fallback, feedback, source = None, None, False, None, "offline-writer"
    max_attempts, attempts_used = (1 if ctx.llm.is_offline else 2), 0
    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        system, user = reporter_prompts(state["question"], facts, feedback)
        try:
            text = ctx.llm.complete(LLMRequest(role="reporter", system=system, user=user, context={"facts": facts}, max_tokens=3000)).strip()
        except Exception as exc:  # noqa: BLE001
            feedback = f"LLM call failed: {type(exc).__name__}: {exc}"
            break
        text = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", text).strip()
        missing = _sections_ok(text)
        grounding = check_numbers(text, facts)
        if not missing and grounding.rate >= MIN_GROUNDING:
            narrative, source = text, ("offline-writer" if ctx.llm.is_offline else "llm")
            break
        problems = []
        if missing:
            problems.append(f"Missing sections: {missing}.")
        if grounding.rate < MIN_GROUNDING:
            problems.append(f"These numbers do not appear in FACTS: {grounding.ungrounded[:12]}. Use only numbers from FACTS, exactly as written there.")
        feedback = " ".join(problems)
    if narrative is None:
        narrative, used_fallback, source = offline.write_narrative(facts), True, "deterministic-fallback"
        grounding = check_numbers(narrative, facts)

    # ---- assemble + write artefacts ----------------------------------------------------------
    chart_files = [c for c in (stats or {}).get("charts", [])] + [
        f for f in ("model_feature_importance.png", "roc_curve.png", "pred_vs_actual.png") if (art / "charts" / f).is_file()
    ]
    agents = {k: dict(v) for k, v in state.get("agents", {}).items()}
    fallbacks = [name for name, st in agents.items() if st.get("used_fallback")] + (["reporter"] if used_fallback else [])
    meta = {
        "run_id": ctx.run_dir.name, "generated_at": utc_now(), "llm": ctx.llm.name, "sandbox": ctx.sandbox.mode,
        "attempts": {k: v.get("attempts", 0) for k, v in agents.items()}, "fallbacks": fallbacks,
        "seed": ctx.settings.seed, "grounding": grounding.to_dict(), "python": platform.python_version(),
    }  # fmt: skip
    report_md = assemble_report(narrative=narrative, facts=facts, cleaning=cleaning, stats=stats, metrics=metrics,
                                chart_files=list(dict.fromkeys(chart_files)), meta=meta)  # fmt: skip
    (art / "report.md").write_text(report_md, encoding="utf-8")
    (art / "report.html").write_text(to_html(report_md, art), encoding="utf-8")
    (art / "facts.json").write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    (art / "grounding.json").write_text(json.dumps(grounding.to_dict(), indent=2), encoding="utf-8")

    agents["reporter"] = {"attempts": attempts_used, "status": "ok", "used_fallback": used_fallback, "last_error": feedback if used_fallback else None}
    event = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"), "agent": "reporter", "status": "ok", "attempt": 1,
        "source": source, "message": f"reporter wrote the report ({source}; {grounding.grounded}/{grounding.checked} numbers verified)",
    }  # fmt: skip
    ctx.emit(event)
    return {"facts": facts, "narrative": narrative, "grounding": grounding.to_dict(), "agents": agents, "events": [event]}
