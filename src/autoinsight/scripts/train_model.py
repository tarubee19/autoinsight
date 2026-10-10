# Trusted modelling script (NOT LLM-written): trains, validates and packages the predictive model.
#
# Contract
#   inputs : INPUT_PATH (cleaned CSV), PARAMS (plan incl. seed)
#   outputs: model.joblib, metrics.json, feature_importance.json, schema.json,
#            charts/roc_curve.png | charts/pred_vs_actual.png, charts/model_feature_importance.png
#
# Design choices worth knowing:
#   * A held-out 20 % test set is split off first and touched exactly once, for the final numbers.
#   * Model selection (baseline vs logistic/ridge vs gradient boosting) uses cross-validation on the
#     training split only.
#   * The decision threshold is chosen on out-of-fold predictions (never on the test set).
#   * Importance = permutation importance on the test set, computed on the *raw* columns, so
#     one-hot-encoded levels are automatically grouped back into their original feature.
#   * Everything stored in model.joblib is a plain scikit-learn object + dicts (no custom classes),
#     so the API can load it without importing this script.
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)  # fmt: skip
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = PARAMS["target"]
TASK = PARAMS["task_type"]
SEED = int(PARAMS.get("seed", 42))
INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
})

df = pd.read_csv(INPUT_PATH, keep_default_na=False, na_values=[""])
y = df[TARGET]
X = df.drop(columns=[TARGET])
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]
for c in cat_cols:
    X[c] = X[c].astype(str)
warnings_out = []


def preprocessor(scale):
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, num_cols),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


# ------------------------------------------------------------------ split + candidates
classification = TASK == "classification"
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y if classification else None
)
n_folds = 3 if len(X_train) > 50_000 else 5
if classification:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scoring = "roc_auc"
    candidates = {
        "baseline (majority class)": Pipeline([("prep", preprocessor(False)), ("model", DummyClassifier(strategy="prior"))]),
        "logistic regression": Pipeline([("prep", preprocessor(True)), ("model", LogisticRegression(max_iter=2000, C=0.5))]),
        "gradient boosting": Pipeline([
            ("prep", preprocessor(False)),
            ("model", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, l2_regularization=1.0, early_stopping=True, random_state=SEED)),
        ]),
    }  # fmt: skip
else:
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scoring = "r2"
    candidates = {
        "baseline (mean)": Pipeline([("prep", preprocessor(False)), ("model", DummyRegressor())]),
        "ridge regression": Pipeline([("prep", preprocessor(True)), ("model", Ridge(alpha=1.0))]),
        "gradient boosting": Pipeline([
            ("prep", preprocessor(False)),
            ("model", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, l2_regularization=1.0, early_stopping=True, random_state=SEED)),
        ]),
    }  # fmt: skip

comparison = []
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
    comparison.append({"model": name, "cv_mean": float(scores.mean()), "cv_std": float(scores.std())})
    log(f"cv {name}: {scores.mean():.4f} +/- {scores.std():.4f}")
best_name = max((c for c in comparison if not c["model"].startswith("baseline")), key=lambda c: c["cv_mean"])["model"]
best = candidates[best_name]

# ------------------------------------------------------------------ threshold from out-of-fold predictions
threshold = 0.5
if classification:
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    threshold = float(thresholds[int(np.argmax(f1))])

best.fit(X_train, y_train)

# ------------------------------------------------------------------ final test metrics
metrics = {
    "task_type": TASK,
    "model": best_name,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "cv_folds": n_folds,
    "cv_metric": scoring,
    "model_comparison": comparison,
}
if classification:
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics.update(
        threshold=threshold,
        roc_auc=float(roc_auc_score(y_test, proba)),
        pr_auc=float(average_precision_score(y_test, proba)),
        accuracy=float(accuracy_score(y_test, pred)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
        brier=float(brier_score_loss(y_test, proba)),
        positive_rate_test=float(y_test.mean()),
        baseline_accuracy=float(max(y_test.mean(), 1 - y_test.mean())),
    )
    # Top-decile lift: how much better than random is the 10 % of rows the model ranks highest?
    order = np.argsort(-proba)
    top_n = max(1, int(0.1 * len(order)))
    metrics["lift_top_decile"] = float(y_test.to_numpy()[order[:top_n]].mean() / max(y_test.mean(), 1e-9))
    metrics["capture_top_decile"] = float(y_test.to_numpy()[order[:top_n]].sum() / max(y_test.sum(), 1))
    bins = pd.cut(pd.Series(proba), bins=[0, 0.1, 0.2, 0.4, 0.6, 1.0], include_lowest=True)
    calibration = pd.DataFrame({"bin": bins.astype(str), "pred": proba, "obs": y_test.to_numpy()}).groupby("bin", observed=True).agg(
        n=("obs", "size"), mean_predicted=("pred", "mean"), observed_rate=("obs", "mean")
    ).reset_index()
    metrics["calibration_bins"] = calibration.to_dict(orient="records")
    if metrics["roc_auc"] > 0.99:
        warnings_out.append("Test ROC AUC above 0.99 - check for target leakage (a feature derived from the outcome).")
else:
    pred = best.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    metrics.update(
        r2=float(r2_score(y_test, pred)),
        mae=float(mean_absolute_error(y_test, pred)),
        rmse=rmse,
        target_std_test=float(y_test.std()),
    )
    if metrics["r2"] > 0.99:
        warnings_out.append("Test R2 above 0.99 - check for target leakage.")

# ------------------------------------------------------------------ leakage scan (single-feature power)
suspects = []
if classification:
    for col in X.columns:
        try:
            if col in num_cols:
                auc = roc_auc_score(y, X[col])
            else:
                rate = y.groupby(X[col]).transform("mean")
                auc = roc_auc_score(y, rate)
            auc = max(auc, 1 - auc)
            if auc > 0.97:
                suspects.append({"feature": col, "single_feature_auc": float(auc)})
        except ValueError:
            continue
    for item in suspects:
        warnings_out.append(f"Feature '{item['feature']}' alone reaches AUC {item['single_feature_auc']:.2f} - possible leakage.")
metrics["warnings"] = warnings_out

# ------------------------------------------------------------------ permutation importance on raw columns
sample = X_test if len(X_test) <= 4000 else X_test.sample(4000, random_state=SEED)
imp = permutation_importance(
    best, sample, y_test.loc[sample.index], scoring="roc_auc" if classification else "r2",
    n_repeats=8, random_state=SEED, n_jobs=1,
)
fitted_pred = best.predict_proba(sample)[:, 1] if classification else best.predict(sample)
rows = []
total = float(np.clip(imp.importances_mean, 0, None).sum()) or 1.0
for i, col in enumerate(sample.columns):
    row = {
        "feature": col,
        "kind": "numeric" if col in num_cols else "categorical",
        "importance": float(imp.importances_mean[i]),
        "importance_std": float(imp.importances_std[i]),
        "share_of_total": float(max(imp.importances_mean[i], 0) / total),
    }
    if col in num_cols:
        corr = pd.Series(sample[col].to_numpy(dtype=float)).corr(pd.Series(fitted_pred), method="spearman")
        row["effect_direction"] = "higher value -> higher prediction" if (corr or 0) >= 0 else "higher value -> lower prediction"
        row["effect_spearman"] = None if corr != corr else float(corr)
    else:
        by_level = pd.Series(fitted_pred, index=sample.index).groupby(sample[col]).agg(["mean", "count"])
        by_level = by_level[by_level["count"] >= 20]
        if len(by_level):
            row["highest_level"] = str(by_level["mean"].idxmax())
            row["lowest_level"] = str(by_level["mean"].idxmin())
    rows.append(row)
rows.sort(key=lambda r: -r["importance"])
save_json({"metric": "roc_auc drop" if classification else "r2 drop", "features": rows}, "feature_importance.json")

# ------------------------------------------------------------------ charts
top = rows[:12][::-1]
fig, ax = plt.subplots(figsize=(7.2, 0.4 * len(top) + 1.3))
ax.barh([r["feature"][:28] for r in top], [max(r["importance"], 0) for r in top], xerr=[r["importance_std"] for r in top],
        color=BLUE, height=0.55, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})  # fmt: skip
ax.set_xlabel(f"Drop in test {'ROC AUC' if classification else 'R2'} when the feature is shuffled")
ax.set_title("What the model relies on (permutation importance, held-out data)")
ax.grid(axis="y", visible=False)
save_fig(fig, "model_feature_importance.png")

if classification:
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve on held-out data (AUC = {metrics['roc_auc']:.3f})")
    ax.set_aspect("equal")
    save_fig(fig, "roc_curve.png")
else:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(y_test, pred, s=7, color=BLUE, alpha=0.4, edgecolors="none")
    lims = [float(min(y_test.min(), pred.min())), float(max(y_test.max(), pred.max()))]
    ax.plot(lims, lims, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs actual (R2 = {metrics['r2']:.3f})")
    save_fig(fig, "pred_vs_actual.png")

# ------------------------------------------------------------------ schema for the prediction API / UI
schema_features = []
for col in X.columns:
    if col in num_cols:
        series = X[col]
        low_card = series.nunique() <= 10
        entry = {
            "name": col, "kind": "numeric", "min": float(series.min()), "max": float(series.max()),
            "median": float(series.median()), "mean": float(series.mean()),
            "integer": bool(np.allclose(series, series.round())),
            "choices": sorted(float(v) for v in series.unique()) if low_card else None,
            "default": float(series.median()),
        }  # fmt: skip
    else:
        counts = X[col].value_counts()
        entry = {"name": col, "kind": "categorical", "choices": [str(v) for v in counts.index[:50]], "default": str(counts.index[0])}
    schema_features.append(entry)
schema = {
    "target": TARGET,
    "task_type": TASK,
    "positive_label": PARAMS.get("positive_label"),
    "model": best_name,
    "threshold": threshold if classification else None,
    "base_rate": float(y.mean()) if classification else None,
    "features": schema_features,
}
save_json(schema, "schema.json")

# Per-feature samples of observed values (NOT linked rows) so the API can explain a single prediction
# as "this value vs the average value of that feature". 40 cells per feature, frequency-weighted.
marginals = {col: X[col].sample(40, random_state=SEED, replace=len(X) < 40).tolist() for col in X.columns}

# Refit on ALL data for the shipped model (the metrics above are from the honest split).
best.fit(X, y)
joblib.dump(
    {
        "pipeline": best,
        "feature_order": list(X.columns),
        "marginals": marginals,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "task_type": TASK,
        "threshold": threshold if classification else None,
        "base_rate": float(y.mean()) if classification else None,
        "positive_label": PARAMS.get("positive_label"),
        "target": TARGET,
    },
    os.path.join(OUT_DIR, "model.joblib"),
)
save_json(metrics, "metrics.json")
log(f"trained {best_name}; " + (f"test ROC AUC {metrics['roc_auc']:.3f}" if classification else f"test R2 {metrics['r2']:.3f}"))
