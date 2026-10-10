import pandas as pd
import pytest

from autoinsight.profiling import (
    PlanError,
    build_heuristic_plan,
    detect_target,
    load_dataframe,
    normalise_plan,
    profile_dataframe,
)


def plan_for(df, question, hint=None):
    return build_heuristic_plan(df, profile_dataframe(df), question, hint)


def test_target_is_found_from_the_question(samples):
    df = load_dataframe(samples / "telco_churn.csv")
    plan = plan_for(df, "Find the driving factors behind customer churn")
    assert plan["target"] == "Churn" and plan["task_type"] == "classification"
    assert plan["positive_label"] == "Yes" and plan["negative_label"] == "No"
    assert plan["id_columns"] == ["customerID"]


def test_id_like_columns_are_never_the_target(samples):
    df = load_dataframe(samples / "hr_attrition.csv")
    # "employees" matches the column EmployeeNumber by name, but that is an identifier
    assert detect_target(df, "why do employees leave?") == "Attrition"


@pytest.mark.parametrize("question", ["why do employees leave?", "Why do employees leave the company?", "Find the drivers of attrition"])
def test_outcome_words_map_to_outcome_columns(samples, question):
    # "company" also matches the column YearsAtCompany, but "leave" means attrition
    assert detect_target(load_dataframe(samples / "hr_attrition.csv"), question) == "Attrition"


def test_regression_target_detected(samples):
    plan = plan_for(load_dataframe(samples / "house_prices.csv"), "what drives the sale price of a house")
    assert plan["target"] == "price" and plan["task_type"] == "regression"


def test_ambiguous_question_asks_the_user(samples):
    with pytest.raises(PlanError, match="Mention it in your question"):
        plan_for(load_dataframe(samples / "house_prices.csv"), "analyze the data")


def test_explicit_target_wins_and_must_exist(samples):
    df = load_dataframe(samples / "telco_churn.csv")
    assert plan_for(df, "anything", hint="tenure")["target"] == "tenure"
    with pytest.raises(PlanError, match="not found"):
        plan_for(df, "anything", hint="nope")


def test_multiclass_text_target_is_rejected_clearly():
    df = pd.DataFrame({"colour": ["r", "g", "b"] * 40, "x": range(120)})
    with pytest.raises(PlanError, match="3 classes"):
        plan_for(df, "predict colour", hint="colour")


def test_positive_label_defaults_to_minority_when_no_signal():
    df = pd.DataFrame({"grade": ["pass"] * 90 + ["flag"] * 10, "x": range(100)})
    plan = plan_for(df, "drivers of grade", hint="grade")
    assert plan["positive_label"] == "flag"


def test_leakage_columns_are_dropped_by_name():
    df = pd.DataFrame({"Churn": ["Yes", "No"] * 50, "Churn Reason": ["a", "b"] * 50, "x": range(100)})
    plan = plan_for(df, "churn drivers")
    assert plan["leakage_suspects"] == ["Churn Reason"] and "Churn Reason" in plan["drop_columns"]


def test_none_category_survives_but_placeholder_tokens_become_missing(tmp_path):
    path = tmp_path / "a.csv"
    pd.DataFrame({"svc": ["None", "NA", "DSL", "N/A", "?", " "] * 10, "y": [0, 1] * 30}).to_csv(path, index=False)
    df = load_dataframe(path)
    assert set(df["svc"].dropna().unique()) == {"None", "DSL"}  # "None" is a real category (e.g. 'InternetService: None')
    assert df["svc"].isna().sum() == 40  # "NA", "N/A", "?" and blank are placeholder tokens -> missing


def test_semicolon_files_and_bom_headers_are_read(tmp_path):
    path = tmp_path / "semi.csv"
    path.write_bytes("﻿a;b;c\n1;x;3\n2;y;4\n".encode())
    df = load_dataframe(path)
    assert list(df.columns) == ["a", "b", "c"]


def test_llm_plan_is_validated_against_the_data(samples):
    df = load_dataframe(samples / "telco_churn.csv")
    prof = profile_dataframe(df)
    plan = normalise_plan({"target": "churn", "positive_label": "yes", "drop_columns": ["Ghost", "gender"]}, df, prof, "q", None)
    assert plan["target"] == "Churn" and plan["positive_label"] == "Yes" and plan["drop_columns"] == ["gender"]
    with pytest.raises(PlanError):
        normalise_plan({"target": "does_not_exist"}, df, prof, "q", None)


def test_profile_never_contains_raw_rows_beyond_three_examples(samples):
    df = load_dataframe(samples / "telco_churn.csv")
    prof = profile_dataframe(df)
    assert all(len(c.get("examples", [])) <= 3 for c in prof["columns"])
    assert all("examples" not in c for c in profile_dataframe(df, send_samples=False)["columns"] if c["kind"] not in ("categorical", "boolean"))
