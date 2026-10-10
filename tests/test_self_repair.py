"""The heart of the 'robust agents' claim: failures are caught, explained to the agent, repaired."""

import json

import pytest

from autoinsight import templates
from autoinsight.pipeline import run_pipeline
from autoinsight.sandbox import Sandbox, SandboxUnavailable

from .conftest import make_settings
from .helpers import ANALYST_TEMPLATE, ENGINEER_TEMPLATE, ScriptedLLM, fenced

UNSAFE = fenced("import os\nos.system('echo pwned > /tmp/pwned')")
RUNTIME_BUG = fenced("import pandas as pd\ndf = pd.read_csv(INPUT_PATH)\ndf['x'] = df['NoSuchColumn'] * 2\nsave_csv(df, 'cleaned.csv')")
NO_OUTPUT = fenced("import pandas as pd\ndf = pd.read_csv(INPUT_PATH)\nlog(len(df))")
HALLUCINATED_FEATURE = fenced("save_json({'top_drivers': [{'feature': 'Imaginary', 'strength': 0.9, 'finding': 'x'}], 'base_rate': 0.3}, 'stats.json')")


def run(tmp_path, llm, csv, **settings_kw):
    settings = make_settings(tmp_path, allow_unsafe_local=True, **settings_kw)
    return run_pipeline(csv, "Find the driving factors behind customer churn", settings=settings, llm=llm,
                        sandbox=Sandbox(settings, mode="local"))  # fmt: skip


def test_engineer_recovers_from_unsafe_buggy_and_empty_programs(tmp_path, small_churn):
    llm = ScriptedLLM(engineer=[UNSAFE, RUNTIME_BUG, NO_OUTPUT, ENGINEER_TEMPLATE])
    res = run(tmp_path, llm, small_churn, max_code_attempts=4)
    assert res.ok, res.error
    assert res.summary["attempts"]["engineer"] == 4
    logs = [json.loads(p.read_text()) for p in sorted((res.run_dir / "logs").glob("engineer_attempt*.json"))]
    assert "safety guard" in logs[0]["error"] and "NOT executed" in logs[0]["error"]
    assert "KeyError" in logs[1]["error"] and "line 3" in logs[1]["error"]
    assert "cleaned.csv was not written" in logs[2]["error"]
    assert logs[3]["ok"]


def test_the_error_is_fed_back_to_the_llm(tmp_path, small_churn):
    llm = ScriptedLLM(engineer=[RUNTIME_BUG, ENGINEER_TEMPLATE])
    res = run(tmp_path, llm, small_churn)
    assert res.ok
    repair_prompt = [r for r in llm.requests if r.role == "engineer"][1].user
    assert "YOUR PREVIOUS PROGRAM FAILED" in repair_prompt
    assert "KeyError" in repair_prompt and "NoSuchColumn" in repair_prompt  # code + error both included


def test_analyst_hallucinated_features_are_caught(tmp_path, small_churn):
    llm = ScriptedLLM(analyst=[HALLUCINATED_FEATURE, ANALYST_TEMPLATE])
    res = run(tmp_path, llm, small_churn)
    assert res.ok and res.summary["attempts"]["analyst"] == 2
    first = json.loads((res.run_dir / "logs" / "analyst_attempt1.json").read_text())
    assert "Imaginary" in first["error"] and "not a column" in first["error"]


def test_falls_back_to_vetted_template_when_the_llm_keeps_failing(tmp_path, small_churn):
    llm = ScriptedLLM(engineer=[RUNTIME_BUG] * 5, analyst=[UNSAFE] * 5)
    res = run(tmp_path, llm, small_churn, max_code_attempts=2)
    assert res.ok, res.error
    assert res.summary["fallbacks"] == ["engineer", "analyst"]
    assert res.summary["attempts"]["engineer"] == 3  # 2 LLM attempts + 1 template
    log = json.loads((res.run_dir / "logs" / "engineer_attempt3.json").read_text())
    assert log["source"] == "template" and log["ok"]
    assert "deterministic fallback used" in res.report_path.read_text().lower()


def test_llm_outage_degrades_gracefully(tmp_path, small_churn):
    boom = ConnectionError("API unreachable")
    llm = ScriptedLLM(planner=[boom, boom], engineer=[boom] * 3, analyst=[boom] * 3, reporter=[boom, boom])
    res = run(tmp_path, llm, small_churn)
    assert res.ok, res.error  # the whole analysis still completes with vetted code
    assert set(res.summary["fallbacks"]) == {"engineer", "analyst", "reporter"}
    assert res.summary["plan_source"] == "heuristic"


def test_reporter_hallucinations_are_rejected_then_corrected(tmp_path, small_churn):
    bad = "## Executive summary\n- Churn is exactly 87.65% and revenue fell 42.1%.\n"
    llm = ScriptedLLM(reporter=[bad])
    res = run(tmp_path, llm, small_churn)
    assert res.ok and res.summary["attempts"]["reporter"] == 2 and res.summary["grounding"]["grounding_rate"] == 1.0
    md = res.report_path.read_text()
    assert "87.65" not in md and "42.1" not in md
    feedback_prompt = [r for r in llm.requests if r.role == "reporter"][1].user
    assert "87.65%" in feedback_prompt and "do not appear in FACTS" in feedback_prompt


def test_reporter_that_never_fixes_itself_is_replaced_by_the_deterministic_writer(tmp_path, small_churn):
    bad = "## Executive summary\n- Churn is exactly 87.65%.\n"
    res = run(tmp_path, ScriptedLLM(reporter=[bad, bad]), small_churn)
    assert res.ok and "reporter" in res.summary["fallbacks"] and "87.65" not in res.report_path.read_text()


def test_llm_plan_with_a_nonexistent_target_falls_back_to_heuristic(tmp_path, small_churn):
    llm = ScriptedLLM(planner=['{"target": "Imaginary"}', "not json at all"])
    res = run(tmp_path, llm, small_churn)
    assert res.ok and res.summary["target"] == "Churn" and res.summary["plan_source"] == "heuristic"


def test_agent_gives_up_cleanly_when_even_the_template_fails(tmp_path, small_churn, monkeypatch):
    original = templates.source
    monkeypatch.setattr(templates, "source", lambda name: "raise RuntimeError('template is broken')\n" if name == templates.ENGINEER else original(name))
    llm = ScriptedLLM(engineer=[RUNTIME_BUG] * 3)
    res = run(tmp_path, llm, small_churn, max_code_attempts=2)
    assert not res.ok and "engineer agent could not produce valid output" in res.error
    assert "could not be completed" in res.report_path.read_text()


def test_llm_code_is_refused_without_a_real_sandbox(tmp_path, small_churn):
    settings = make_settings(tmp_path)  # allow_unsafe_local stays False
    with pytest.raises(SandboxUnavailable):
        run_pipeline(small_churn, "churn", settings=settings, llm=ScriptedLLM(), sandbox=Sandbox(settings, mode="local"))
