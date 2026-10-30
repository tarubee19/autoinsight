"""Real-data benchmark: does AutoInsight hold up outside its own synthetic samples?

Two experiments, both offline and reproducible (seeded), on four public real datasets:

A. END-TO-END. Run the full agent workflow (offline templates) on each raw CSV with a plain-English
   question. Check: did it pick the right target? did cleaning/stats/model/report all finish?
   held-out score of the model it chose, share of report numbers verified, run time.

B. MODEL ZOO. On the *cleaned* data from (A), compare the modeler's candidates with stronger
   off-the-shelf learners using 5-fold x 3-repeat cross-validation (same folds for every model):
   logistic/ridge, random forest, the modeler's HistGradientBoosting config, XGBoost, LightGBM,
   CatBoost, plus a log-target variant for skewed regression targets. Differences are tested with
   the Nadeau-Bengio corrected resampled t-test (plain CV t-tests are overconfident because folds
   share training data). This is what justifies - or refutes - the modeler's candidate list.

Datasets (downloaded from public GitHub mirrors, SHA-256 pinned):
  * IBM Telco customer churn (7,043 x 21)      - classification, "Churn"
  * IBM HR employee attrition (1,470 x 35)     - classification, "Attrition"
  * Ames housing, De Cock 2011 (2,930 x 82)    - regression, "SalePrice"
  * UCI bank marketing, Moro et al. 2014 (41,188 x 21) - classification, "y"

Usage:
    python scripts/benchmark_real_data.py                # writes docs/REAL_DATA_BENCHMARK.md + json
    python scripts/benchmark_real_data.py --skip-e2e     # model zoo only (reuses cleaned data)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # see docs: OpenMP oversubscription made HGB 300x slower

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "real"
DATASETS = {
    "telco_churn_ibm": {
        "url": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "sha256": "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91",
        "question": "Find the driving factors behind customer churn", "target": "Churn",
        "reference": "Published ROC AUC on this dataset is typically 0.84-0.85 (logistic regression and GBMs perform alike).",
    },
    "hr_attrition_ibm": {
        "url": "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv",
        "sha256": "a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659",
        "question": "Why do employees leave the company?", "target": "Attrition",
        "reference": "Small (1,470 rows, 16% positives); linear models usually match or beat GBMs, ROC AUC ~0.80-0.85.",
    },
    "ames_housing": {
        "url": "https://raw.githubusercontent.com/wblakecannon/ames/master/data/housing.csv",
        "sha256": "1cf821e5ab53ce9ac177cb663872c2d2b8c5b385f85eed495ce62a213bd34f86",
        "question": "What drives the sale price of a house?", "target": "SalePrice",
        "reference": "Right-skewed price; GBMs typically reach R2 ~0.90-0.93.",
    },
    "bank_marketing": {
        "url": "https://raw.githubusercontent.com/selva86/datasets/master/bank-full.csv",
        "sha256": "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8",
        "question": "What makes a customer subscribe to a term deposit?", "target": "y",
        "reference": "'duration' (call length) is only known after the call - a classic leakage feature. "
                     "Moro et al. report AUC ~0.80 for a realistic (pre-call) model; with duration ~0.93.",
    },
}


def fetch(name: str) -> Path:
    spec = DATASETS[name]
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.csv"
    if not path.exists():
        urllib.request.urlretrieve(spec["url"], path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise RuntimeError(f"{name}: checksum mismatch ({digest}) - the mirror changed")
    return path


# ----------------------------------------------------------------------------- A. end-to-end
def run_e2e(name: str, workdir: Path) -> dict:
    from autoinsight.config import Settings
    from autoinsight.pipeline import run_pipeline

    spec = DATASETS[name]
    settings = Settings(llm_provider="offline", sandbox_mode="local", runs_dir=workdir / name)
    t0 = time.time()
    res = run_pipeline(fetch(name), spec["question"], settings=settings)
    out = {"seconds": round(time.time() - t0, 1), "status": res.status, "error": res.error,
           "run_dir": str(res.run_dir), "target": res.summary.get("target"), "task": res.summary.get("task_type")}
    art = res.run_dir / "artifacts"
    if (art / "metrics.json").exists():
        m = json.loads((art / "metrics.json").read_text())
        out.update(model=m["model"], score=m.get("roc_auc", m.get("r2")),
                   metric="ROC AUC" if "roc_auc" in m else "R2", warnings=m.get("warnings", []),
                   cv_comparison=m.get("model_comparison"))
    g = res.summary.get("grounding") or {}
    out["grounding_rate"], out["numbers_checked"] = g.get("grounding_rate"), g.get("numbers_checked")
    out["target_correct"] = out["target"] == spec["target"]
    out["attempts"], out["fallbacks"] = res.summary.get("attempts"), res.summary.get("fallbacks")
    return out


# ----------------------------------------------------------------------------- B. model zoo
def model_zoo(task: str, num_cols: list[str], cat_cols: list[str], seed: int = 42) -> dict:
    from catboost import CatBoostClassifier, CatBoostRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from xgboost import XGBClassifier, XGBRegressor

    def ohe(scale):
        return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", num_cols),
                                  ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                                        sparse_output=False), cat_cols)])

    def ordinal():
        return ColumnTransformer([("num", "passthrough", num_cols),
                                  ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                         encoded_missing_value=-1), cat_cols)])

    n_cat = len(cat_cols)
    cat_idx = list(range(len(num_cols), len(num_cols) + n_cat))
    hgb_kw = dict(max_iter=250, learning_rate=0.06, l2_regularization=1.0, early_stopping=True, random_state=seed)
    gbm_kw = dict(n_estimators=400, learning_rate=0.05, random_state=seed)
    if task == "classification":
        return {
            "baseline (prior)": Pipeline([("p", ohe(False)), ("m", DummyClassifier(strategy="prior"))]),
            "logistic regression (current)": Pipeline([("p", ohe(True)), ("m", LogisticRegression(max_iter=3000, C=0.5))]),
            "random forest": Pipeline([("p", ordinal()), ("m", RandomForestClassifier(500, min_samples_leaf=3, n_jobs=1, random_state=seed))]),
            "HistGB (current)": Pipeline([("p", ohe(False)), ("m", HistGradientBoostingClassifier(**hgb_kw))]),
            "XGBoost": Pipeline([("p", ordinal()), ("m", XGBClassifier(**gbm_kw, max_depth=4, subsample=0.8, colsample_bytree=0.8, n_jobs=1, verbosity=0))]),
            "LightGBM": Pipeline([("p", ordinal()), ("m", LGBMClassifier(**gbm_kw, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1, verbose=-1))]),
            "CatBoost": CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, cat_features=cat_cols, random_seed=seed, verbose=0, thread_count=1),
        }
    return {
        "baseline (mean)": Pipeline([("p", ohe(False)), ("m", DummyRegressor())]),
        "ridge regression (current)": Pipeline([("p", ohe(True)), ("m", Ridge(alpha=1.0))]),
        "ridge, log target": TransformedTargetRegressor(Pipeline([("p", ohe(True)), ("m", Ridge(alpha=1.0))]), func=np.log1p, inverse_func=np.expm1),
        "random forest": Pipeline([("p", ordinal()), ("m", RandomForestRegressor(500, min_samples_leaf=2, n_jobs=1, random_state=seed))]),
        "HistGB (current)": Pipeline([("p", ohe(False)), ("m", HistGradientBoostingRegressor(**hgb_kw))]),
        "HistGB, log target": TransformedTargetRegressor(Pipeline([("p", ohe(False)), ("m", HistGradientBoostingRegressor(**hgb_kw))]), func=np.log1p, inverse_func=np.expm1),
        "XGBoost": Pipeline([("p", ordinal()), ("m", XGBRegressor(**gbm_kw, max_depth=4, subsample=0.8, colsample_bytree=0.8, n_jobs=1, verbosity=0))]),
        "LightGBM": Pipeline([("p", ordinal()), ("m", LGBMRegressor(**gbm_kw, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1, verbose=-1))]),
        "CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, cat_features=cat_cols, random_seed=seed, verbose=0, thread_count=1),
    }


def corrected_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> float:
    """Nadeau & Bengio (2003) corrected resampled t-test, two-sided p-value."""
    k = len(diffs)
    var = diffs.var(ddof=1)
    if var == 0:
        return 1.0
    t = diffs.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(2 * stats.t.sf(abs(t), df=k - 1))


def run_zoo(name: str, cleaned: Path, target: str, task: str, drop: list[str] | None = None,
            seed: int = 42) -> dict:
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score, r2_score, roc_auc_score, mean_absolute_error
    from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold

    df = pd.read_csv(cleaned, keep_default_na=False, na_values=[""])
    y = df.pop(target)
    if drop:
        df = df.drop(columns=[c for c in drop if c in df.columns])
    if task == "classification":
        pos = sorted(y.astype(str).unique(), key=lambda v: (y.astype(str) == v).mean())[0]
        y = (y.astype(str) == pos).astype(int)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    for c in cat_cols:
        df[c] = df[c].astype(str)
    X = df[num_cols + cat_cols]
    models = model_zoo(task, num_cols, cat_cols, seed)
    splitter = (RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed) if task == "classification"
                else RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed))
    folds = list(splitter.split(X, y))
    res = {}
    for mname, model in models.items():
        scores, extra, fit_s = [], [], []
        for tr, te in folds:
            m = model_zoo(task, num_cols, cat_cols, seed)[mname]   # fresh object (CatBoost cannot be cloned)
            t0 = time.time()
            m.fit(X.iloc[tr], y.iloc[tr])
            fit_s.append(time.time() - t0)
            if task == "classification":
                p = m.predict_proba(X.iloc[te])[:, 1]
                scores.append(roc_auc_score(y.iloc[te], p))
                extra.append(average_precision_score(y.iloc[te], p))
            else:
                p = m.predict(X.iloc[te])
                scores.append(r2_score(y.iloc[te], p))
                extra.append(mean_absolute_error(y.iloc[te], p))
        res[mname] = {"scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores)),
                      "secondary": float(np.mean(extra)), "fit_s": float(np.mean(fit_s))}
        print(f"  {name:18s} {mname:30s} {np.mean(scores):.4f} +/- {np.std(scores):.4f}  fit {np.mean(fit_s):.2f}s",
              flush=True)
    n_te = len(folds[0][1])
    n_tr = len(folds[0][0])
    current = [k for k in res if "(current)" in k]
    ref = max(current, key=lambda k: res[k]["mean"])
    for k, v in res.items():
        v["vs_current_best"] = None if k == ref else {
            "diff": float(np.mean(np.array(v["scores"]) - np.array(res[ref]["scores"]))),
            "p_value": corrected_ttest(np.array(v["scores"]) - np.array(res[ref]["scores"]), n_tr, n_te)}
    for v in res.values():
        v["fold_scores"] = [round(x, 5) for x in v.pop("scores")]
    return {"rows": len(X), "features": X.shape[1], "metric": "ROC AUC" if task == "classification" else "R2",
            "secondary_metric": "PR AUC" if task == "classification" else "MAE",
            "reference_model": ref, "models": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    ap.add_argument("--workdir", type=Path, default=ROOT / "runs" / "real_benchmark")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--e2e-only", action="store_true",
                    help="re-run only the end-to-end runs; keep the model-zoo numbers from the existing json")
    a = ap.parse_args()
    out_json = a.out / "real_data_benchmark.json"
    previous = json.loads(out_json.read_text()) if out_json.exists() else {}
    results = dict(previous) if a.e2e_only else {}
    for name in a.datasets:
        spec = DATASETS[name]
        print(f"== {name}", flush=True)
        e2e = run_e2e(name, a.workdir)
        print("  e2e:", {k: e2e.get(k) for k in ("target", "model", "score", "seconds", "numbers_verified")}, flush=True)
        cleaned = Path(e2e["run_dir"]) / "artifacts" / "cleaned.csv"
        if a.e2e_only and name in previous:
            results[name] = {**previous[name], "e2e": e2e}
            out_json.write_text(json.dumps(results, indent=2, default=str))
            continue
        zoo = run_zoo(name, cleaned, spec["target"], e2e["task"]) if cleaned.exists() else None
        results[name] = {"e2e": e2e, "zoo": zoo, "reference": spec["reference"]}
        if name == "bank_marketing" and cleaned.exists():   # leakage experiment
            results[name]["zoo_without_duration"] = run_zoo(name, cleaned, spec["target"], e2e["task"], drop=["duration"])
        out_json.write_text(json.dumps(results, indent=2, default=str))
    print("saved", out_json)


if __name__ == "__main__":
    main()
