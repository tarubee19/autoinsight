"""Messy-data torture tests: the pipeline must either succeed with sensible output or fail with a
clear, user-facing message - never crash, never silently produce nonsense."""

import json

import numpy as np
import pandas as pd
import pytest

from autoinsight.pipeline import run_pipeline

from .conftest import SAMPLES, make_settings

Q = "Find the driving factors behind customer churn"


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    df = pd.read_csv(SAMPLES / "telco_churn.csv").drop_duplicates().head(700)
    return df.reset_index(drop=True)


def run(tmp_root, name, df, question=Q, **csv_kw):
    path = tmp_root / f"{name}.csv"
    csv_kw.setdefault("index", False)
    df.to_csv(path, **csv_kw)
    return run_pipeline(path, question, settings=make_settings(tmp_root / f"rob_{name}"))


def cleaned(res) -> pd.DataFrame:
    return pd.read_csv(res.run_dir / "artifacts" / "cleaned.csv", keep_default_na=False, na_values=[""])


def assert_sound(res):
    assert res.ok, res.error
    df = cleaned(res)
    assert not df.isna().any().any(), "cleaned data must have no missing values"
    assert set(df["Churn"].unique()) == {0, 1}
    stats = json.loads((res.run_dir / "artifacts" / "stats.json").read_text())
    assert "Contract" in [d["feature"] for d in stats["top_drivers"][:4]]
    assert json.loads((res.run_dir / "artifacts" / "grounding.json").read_text())["grounding_rate"] >= 0.9


def test_semicolon_delimiter_bom_and_shuffled_columns(tmp_root, base):
    df = base[list(np.random.default_rng(1).permutation(base.columns))]
    assert_sound(run(tmp_root, "semi", df, sep=";", encoding="utf-8-sig"))


def test_heavy_missingness(tmp_root, base):
    df = base.copy()
    rng = np.random.default_rng(2)
    for col in ("tenure", "MonthlyCharges", "Contract", "InternetService", "TechSupport"):
        df.loc[rng.random(len(df)) < 0.35, col] = np.nan
    assert_sound(run(tmp_root, "missing", df))


@pytest.mark.parametrize(
    "mapping",
    [{"Yes": 1, "No": 0}, {"Yes": True, "No": False}, {"Yes": "churned", "No": "retained"}, {"Yes": "Y", "No": "N"}],
    ids=["ints", "bools", "words", "letters"],
)
def test_target_encodings(tmp_root, base, mapping):
    df = base.copy()
    df["Churn"] = df["Churn"].map(mapping)
    res = run(tmp_root, f"target_{list(map(str, mapping.values()))[0]}", df)
    assert_sound(res)
    plan_label = json.loads((res.run_dir / "artifacts" / "cleaning_log.json").read_text())["target"]["positive_label"]
    assert plan_label.lower() in {"1", "true", "churned", "y"}


def test_date_columns_become_features_and_free_text_is_dropped(tmp_root, base):
    df = base.copy()
    df["signup_date"] = pd.date_range("2021-01-01", periods=len(df), freq="D").strftime("%Y-%m-%d")
    df["notes"] = [f"customer note number {i} with unique words {i * 7919}" for i in range(len(df))]
    df["row_uuid"] = [f"{i:08x}-aaaa-bbbb" for i in range(len(df))]
    res = run(tmp_root, "dates", df)
    assert_sound(res)
    cols = set(cleaned(res).columns)
    assert {"signup_date_year", "signup_date_month", "signup_date_dow"} <= cols
    assert not ({"signup_date", "notes", "row_uuid"} & cols)


def test_leakage_column_is_excluded_from_the_model(tmp_root, base):
    df = base.copy()
    df["Churn Reason"] = np.where(df["Churn"] == "Yes", "left for competitor", "n/a")
    res = run(tmp_root, "leak", df)
    assert_sound(res)
    assert "Churn Reason" not in cleaned(res).columns
    assert json.loads((res.run_dir / "artifacts" / "metrics.json").read_text())["roc_auc"] < 0.95


def test_constant_and_all_missing_columns_are_dropped(tmp_root, base):
    df = base.copy()
    df["always"] = "x"
    df["empty"] = np.nan
    res = run(tmp_root, "const", df)
    assert_sound(res)
    assert not ({"always", "empty"} & set(cleaned(res).columns))


def test_none_category_is_kept_as_a_value(tmp_root, base):
    df = base.copy()
    df["InternetService"] = df["InternetService"].replace({"No": "None"})
    res = run(tmp_root, "nonecat", df)
    assert_sound(res)
    assert "None" in set(cleaned(res)["InternetService"])


def test_tiny_dataset_still_works(tmp_root, base):
    df = pd.concat([base[base.Churn == "Yes"].head(30), base[base.Churn == "No"].head(60)])
    assert_sound(run(tmp_root, "tiny", df))


def test_unicode_names_and_values(tmp_root, base):
    df = base.rename(columns={"InternetService": "Internet Servicé (€)", "PaymentMethod": "Zahlungsart"})
    df["Zahlungsart"] = df["Zahlungsart"].fillna("Überweisung")
    assert_sound(run(tmp_root, "unicode", df))


def test_regression_target_with_outliers_and_text_numbers(tmp_root):
    df = pd.read_csv(SAMPLES / "house_prices.csv").head(800)
    df.loc[:5, "price"] = df["price"].max() * 50
    df["sqft"] = df["sqft"].map(lambda v: f"{v:,.0f} sqft".replace(" sqft", ""))  # "1,900" stored as text
    res = run(tmp_root, "reg", df, question="what drives the sale price of a house")
    assert res.ok, res.error
    assert res.summary["task_type"] == "regression" and "sqft" in res.summary["top_drivers"][:3]


def test_multiclass_target_fails_with_a_helpful_message(tmp_root, base):
    df = base.copy()
    df["Churn"] = np.random.default_rng(3).choice(["low", "medium", "high"], len(df))
    res = run(tmp_root, "multi", df)
    assert not res.ok and "3 classes" in res.error and "binary classification" in res.error


def test_target_with_single_value_fails_cleanly(tmp_root, base):
    df = base.copy()
    df["Churn"] = "No"
    res = run(tmp_root, "single", df)
    assert not res.ok and ("single value" in res.error or "Could not tell" in res.error)
