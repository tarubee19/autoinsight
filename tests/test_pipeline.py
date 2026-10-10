import json

import pytest

from autoinsight.pipeline import run_pipeline

from .conftest import make_settings

EXPECTED_ARTIFACTS = {
    "report.md", "report.html", "cleaned.csv", "cleaning_log.json", "stats.json", "metrics.json",
    "model.joblib", "schema.json", "feature_importance.json", "facts.json", "grounding.json",
}  # fmt: skip


def test_full_run_produces_every_artifact(churn_run):
    assert churn_run.ok, churn_run.error
    produced = set(churn_run.summary["artifacts"])
    assert EXPECTED_ARTIFACTS <= produced
    assert {"charts/driver_strength.png", "charts/roc_curve.png", "charts/categorical_drivers.png"} <= produced


def test_churn_drivers_are_recovered(churn_run):
    top5 = churn_run.summary["top_drivers"]
    planted = {"Contract", "tenure", "InternetService", "TechSupport", "MonthlyCharges"}
    assert len(planted & set(top5)) >= 4, top5
    assert top5[0] == "Contract"


def test_model_beats_baseline_and_is_honest(churn_run):
    metrics = json.loads((churn_run.run_dir / "artifacts" / "metrics.json").read_text())
    assert metrics["roc_auc"] > 0.80
    assert metrics["warnings"] == []
    baseline = next(c for c in metrics["model_comparison"] if c["model"].startswith("baseline"))
    assert baseline["cv_mean"] == pytest.approx(0.5, abs=0.01)
    assert metrics["lift_top_decile"] > 2.0


def test_cleaning_fixed_the_planted_mess(churn_run):
    log = json.loads((churn_run.run_dir / "artifacts" / "cleaning_log.json").read_text())
    assert log["duplicates_removed"] == 40
    assert "customerID" in log["dropped_columns"] and "Country" in log["dropped_columns"]
    assert "TotalCharges" in log["coerced_to_numeric"]          # numbers stored as text
    assert log["clipped_outliers"].get("MonthlyCharges", 0) >= 10  # the 9999 entry errors
    assert log["target"]["positive_label"] == "Yes"


def test_report_numbers_are_all_traceable(churn_run):
    grounding = json.loads((churn_run.run_dir / "artifacts" / "grounding.json").read_text())
    assert grounding["grounding_rate"] == 1.0 and grounding["numbers_checked"] > 30


def test_report_structure(churn_run):
    md = churn_run.report_path.read_text()
    for heading in ("## Executive summary", "## Data quality", "## What drives the outcome", "## Predictive model",
                    "## Recommendations", "## Limitations", "## Charts", "## Appendix A", "## Appendix D"):  # fmt: skip
        assert heading in md
    assert "![" in md and "charts/driver_strength.png" in md
    html = (churn_run.run_dir / "artifacts" / "report.html").read_text()
    assert "data:image/png;base64," in html and "<table>" in html


def test_state_and_logs_are_written(churn_run):
    state = json.loads((churn_run.run_dir / "state.json").read_text())
    assert state["status"] == "completed" and set(state["stages"].values()) == {"done"}
    assert (churn_run.run_dir / "logs" / "engineer_attempt1.py").is_file()
    assert not (churn_run.run_dir / "work").exists()  # scratch space cleaned up


def test_regression_run(tmp_root, samples):
    res = run_pipeline(samples / "house_prices.csv", "what drives the sale price of a house", settings=make_settings(tmp_root / "reg"))
    assert res.ok, res.error
    assert res.summary["task_type"] == "regression"
    assert res.summary["model"]["r2"] > 0.85
    assert {"sqft", "neighborhood", "year_built"} <= set(res.summary["top_drivers"])


def test_unknown_target_fails_cleanly_with_a_report(tmp_root, samples):
    res = run_pipeline(samples / "telco_churn.csv", "anything", target="nope", settings=make_settings(tmp_root / "bad"))
    assert not res.ok and "not found" in res.error
    assert "could not be completed" in res.report_path.read_text()
    assert json.loads((res.run_dir / "state.json").read_text())["status"] == "failed"


def test_events_are_streamed(tmp_root, small_churn):
    seen = []
    res = run_pipeline(small_churn, "drivers of churn", settings=make_settings(tmp_root / "ev"), on_event=seen.append)
    assert res.ok
    assert [e["agent"] for e in seen] == ["profile", "planner", "engineer", "analyst", "modeler", "reporter", "finalize"]
