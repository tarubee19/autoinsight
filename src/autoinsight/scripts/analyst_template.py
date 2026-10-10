# Vetted baseline for the Analyst agent (statistical tests + charts).
#
# Contract
#   inputs : INPUT_PATH (cleaned CSV from the Data Engineer), PARAMS (plan)
#   outputs: stats.json, charts/*.png
#
# Method: every feature is tested against the target with a test suited to its type, effect sizes
# are put on one 0-1 "strength" scale so numeric and categorical drivers can be ranked together,
# and p-values are corrected for multiple testing (Benjamini-Hochberg).

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

TARGET = PARAMS["target"]
TASK = PARAMS["task_type"]
POS_NAME = f"{TARGET} = {PARAMS.get('positive_label', 1)}"
NEG_NAME = f"{TARGET} = {PARAMS.get('negative_label', 0)}"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
})


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def strength_label(value):
    if value < 0.1:
        return "negligible"
    if value < 0.3:
        return "small"
    if value < 0.5:
        return "medium"
    return "large"


def short(text, n=22):
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "..."


df = pd.read_csv(INPUT_PATH, keep_default_na=False, na_values=[""])
y = df[TARGET]
X = df.drop(columns=[TARGET])

num_cols, cat_cols, skipped = [], [], []
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 10:
        num_cols.append(col)
    else:
        X[col] = X[col].astype(str)
        if X[col].nunique() < 2:
            skipped.append({"feature": col, "reason": "constant"})
        elif X[col].nunique() > 30:
            skipped.append({"feature": col, "reason": "more than 30 categories"})
        else:
            cat_cols.append(col)

records = []  # one entry per tested feature, unified scale

if TASK == "classification":
    is_pos = (y == 1).to_numpy()
    base_rate = float(is_pos.mean())
    for col in num_cols:
        a = X.loc[is_pos, col].to_numpy(dtype=float)
        b = X.loc[~is_pos, col].to_numpy(dtype=float)
        if len(a) < 5 or len(b) < 5 or np.std(X[col].to_numpy(dtype=float)) == 0:
            skipped.append({"feature": col, "reason": "too little variation"})
            continue
        u_stat, p_mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
        _, p_welch = stats.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
        strength = abs(d) / np.sqrt(d * d + 4)  # Cohen's d -> correlation-scale effect size
        direction = "higher" if d > 0 else "lower"
        records.append({
            "feature": col, "kind": "numeric", "test": "Mann-Whitney U", "p_value": float(p_mwu),
            "p_welch": float(p_welch), "cohens_d": d, "auc": float(u_stat / (len(a) * len(b))),
            "strength": float(strength), "direction": direction,
            "median_positive": float(np.median(a)), "median_negative": float(np.median(b)),
            "mean_positive": float(a.mean()), "mean_negative": float(b.mean()),
            "finding": (
                f"{col} is {direction} when {POS_NAME} (median {np.median(a):.2f} vs {np.median(b):.2f}; "
                f"Cohen's d = {d:.2f})"
            ),
        })
    for col in cat_cols:
        table = pd.crosstab(X[col], y)
        if table.shape[1] < 2 or table.shape[0] < 2:
            skipped.append({"feature": col, "reason": "no contrast"})
            continue
        chi2, p_value, _, expected = stats.chi2_contingency(table, correction=False)
        n_total = table.to_numpy().sum()
        cramers_v = float(np.sqrt(chi2 / (n_total * (min(table.shape) - 1))))
        rates = table[1] / table.sum(axis=1)
        levels = [
            {"level": str(idx), "n": int(table.loc[idx].sum()), "rate": float(rates[idx])}
            for idx in table.index
        ]
        solid = [lv for lv in levels if lv["n"] >= 30] or levels
        top = max(solid, key=lambda lv: lv["rate"])
        low = min(solid, key=lambda lv: lv["rate"])
        records.append({
            "feature": col, "kind": "categorical", "test": "chi-square", "p_value": float(p_value),
            "cramers_v": cramers_v, "strength": cramers_v, "levels": sorted(levels, key=lambda lv: -lv["rate"]),
            "top_level": top["level"], "top_rate": top["rate"], "top_n": top["n"],
            "low_level": low["level"], "low_rate": low["rate"], "low_n": low["n"],
            "low_expected_share": float((expected < 5).mean()),
            "finding": (
                f"{col}: '{top['level']}' has the highest {TARGET} rate ({top['rate']:.1%}, n={top['n']}) "
                f"vs '{low['level']}' at {low['rate']:.1%} (n={low['n']}); overall rate {base_rate:.1%}"
            ),
        })
else:
    base_rate = None
    for col in num_cols:
        if np.std(X[col].to_numpy(dtype=float)) == 0:
            skipped.append({"feature": col, "reason": "no variation"})
            continue
        rho, p_value = stats.spearmanr(X[col], y)
        records.append({
            "feature": col, "kind": "numeric", "test": "Spearman correlation", "p_value": float(p_value),
            "spearman_rho": float(rho), "strength": float(abs(rho)),
            "direction": "higher" if rho > 0 else "lower",
            "finding": f"{col} is {'positively' if rho > 0 else 'negatively'} related to {TARGET} (Spearman rho = {rho:.2f})",
        })
    for col in cat_cols:
        groups = [g.to_numpy(dtype=float) for _, g in y.groupby(X[col]) if len(g) >= 5]
        if len(groups) < 2:
            skipped.append({"feature": col, "reason": "no contrast"})
            continue
        h_stat, p_value = stats.kruskal(*groups)
        grand = y.mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = ((y - grand) ** 2).sum()
        eta = float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0
        means = y.groupby(X[col]).agg(["mean", "count"])
        means = means[means["count"] >= 5]
        top_level, low_level = means["mean"].idxmax(), means["mean"].idxmin()
        records.append({
            "feature": col, "kind": "categorical", "test": "Kruskal-Wallis", "p_value": float(p_value),
            "eta": eta, "strength": eta,
            "levels": [{"level": str(i), "n": int(r["count"]), "mean": float(r["mean"])} for i, r in means.iterrows()],
            "top_level": str(top_level), "low_level": str(low_level),
            "finding": (
                f"{col}: '{top_level}' has the highest mean {TARGET} ({means.loc[top_level, 'mean']:.2f}) "
                f"vs '{low_level}' ({means.loc[low_level, 'mean']:.2f})"
            ),
        })

adjusted = benjamini_hochberg([r["p_value"] for r in records])
for rec, p_adj in zip(records, adjusted, strict=False):
    rec["p_adjusted"] = float(p_adj)
    rec["significant"] = bool(p_adj < 0.05)
    rec["strength_label"] = strength_label(rec["strength"])
records.sort(key=lambda r: (not r["significant"], -r["strength"]))
top_drivers = [
    {k: r[k] for k in ("feature", "kind", "strength", "strength_label", "p_adjusted", "significant", "finding")}
    for r in records[:10]
]

# Redundant numeric features (they would split credit in any model)
multicollinear = []
if len(num_cols) > 1:
    corr_all = X[num_cols].corr(method="spearman")
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            r = corr_all.loc[a, b]
            if abs(r) >= 0.8:
                multicollinear.append({"a": a, "b": b, "spearman_r": float(r)})
    multicollinear.sort(key=lambda m: -abs(m["spearman_r"]))

# ------------------------------------------------------------------ charts
charts = []

# 1. Target distribution
fig, ax = plt.subplots(figsize=(6.2, 3.2))
if TASK == "classification":
    counts = [int((~is_pos).sum()), int(is_pos.sum())]
    bars = ax.bar([NEG_NAME, POS_NAME], counts, color=[BLUE, ORANGE], width=0.5)
    for bar, count in zip(bars, counts, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,}\n({count / len(y):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_ylabel("Rows")
    ax.set_title(f"Target balance: {base_rate:.1%} of rows are {POS_NAME}")
    ax.grid(axis="x", visible=False)
else:
    ax.hist(y, bins=30, color=BLUE, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(TARGET)
    ax.set_ylabel("Rows")
    ax.set_title(f"Distribution of {TARGET} (median {y.median():,.2f})")
save_fig(fig, "target_distribution.png")
charts.append("target_distribution.png")

# 2. Driver strength ranking
shown = records[:10]
if shown:
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(shown) + 1.3))
    labels = [f"{short(r['feature'], 26)} ({'num' if r['kind'] == 'numeric' else 'cat'})" for r in shown][::-1]
    values = [r["strength"] for r in shown][::-1]
    colours = [BLUE if r["significant"] else GRID for r in shown][::-1]
    bars = ax.barh(labels, values, color=colours, height=0.55)
    for bar, value in zip(bars, values, strict=False):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(values) * 1.18 if max(values) > 0 else 1)
    ax.set_xlabel("Association strength with the target (0 = none, 1 = perfect)")
    ax.set_title("Association with the target (grey = not significant)")
    ax.grid(axis="y", visible=False)
    save_fig(fig, "driver_strength.png")
    charts.append("driver_strength.png")

# 3. Top numeric drivers
top_num = [r for r in records if r["kind"] == "numeric"][:3]
if top_num:
    fig, axes = plt.subplots(1, len(top_num), figsize=(4.0 * len(top_num), 3.6))
    axes = np.atleast_1d(axes)
    for ax, rec in zip(axes, top_num, strict=False):
        col = rec["feature"]
        if TASK == "classification":
            data = [X.loc[~is_pos, col].to_numpy(dtype=float), X.loc[is_pos, col].to_numpy(dtype=float)]
            box = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False, medianprops={"color": INK, "linewidth": 1.5})
            for patch, colour in zip(box["boxes"], [BLUE, ORANGE], strict=False):
                patch.set_facecolor(colour)
                patch.set_alpha(0.85)
                patch.set_edgecolor("white")
            ax.set_xticks([1, 2], [short(NEG_NAME, 16), short(POS_NAME, 16)])
            ax.set_title(f"{short(col, 24)}: {rec['direction']} when {short(POS_NAME, 14)}", fontsize=10)
        else:
            sample = df.sample(min(len(df), 1500), random_state=0)
            ax.scatter(sample[col], sample[TARGET], s=6, color=BLUE, alpha=0.35, edgecolors="none")
            ax.set_xlabel(col)
            ax.set_title(f"{short(col, 24)} vs {short(TARGET, 14)} (rho = {rec['spearman_rho']:.2f})", fontsize=10)
        ax.grid(axis="x", visible=False)
    save_fig(fig, "numeric_drivers.png")
    charts.append("numeric_drivers.png")

# 4. Top categorical drivers
top_cat = [r for r in records if r["kind"] == "categorical"][:3]
if top_cat:
    fig, axes = plt.subplots(1, len(top_cat), figsize=(4.4 * len(top_cat), 3.8), sharey=(TASK == "classification"))
    axes = np.atleast_1d(axes)
    key = "rate" if TASK == "classification" else "mean"
    y_top = max(max(lv[key] for lv in (rec["levels"][:8])) for rec in top_cat)
    for ax, rec in zip(axes, top_cat, strict=False):
        levels = [lv for lv in rec["levels"] if lv["n"] >= 30][:8] or rec["levels"][:8]
        names = [short(lv["level"], 14) for lv in levels]
        values = [lv[key] for lv in levels]
        bars = ax.bar(names, values, color=ORANGE if TASK == "classification" else BLUE, width=0.55)
        for bar, value in zip(bars, values, strict=False):
            label = f"{value:.0%}" if TASK == "classification" else f"{value:,.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value, label, ha="center", va="bottom", fontsize=8.5)
        if TASK == "classification":
            ax.axhline(base_rate, color=INK, linestyle="--", linewidth=1)
            ax.text(len(names) - 0.55, base_rate + 0.012 * y_top, f"overall {base_rate:.0%}", va="bottom", ha="right", fontsize=8.5, color=MUTED)
            ax.set_ylabel(f"Share with {POS_NAME}")
            ax.set_ylim(0, min(1.05, y_top * 1.22))
        ax.set_title(short(rec["feature"], 28))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="x", visible=False)
    save_fig(fig, "categorical_drivers.png")
    charts.append("categorical_drivers.png")

# 5. Correlation heatmap (top numeric drivers + target)
heat_cols = [r["feature"] for r in records if r["kind"] == "numeric"][:9]
if len(heat_cols) >= 2:
    frame = X[heat_cols].copy()
    frame[TARGET] = y.to_numpy(dtype=float)
    corr = frame.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.75 * len(corr) + 2.6, 0.7 * len(corr) + 1.8))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), [short(c, 14) for c in corr.columns], rotation=40, ha="right")
    ax.set_yticks(range(len(corr)), [short(c, 14) for c in corr.index])
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(value) > 0.6 else INK)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("Spearman correlation")
    ax.set_title("Correlation between top numeric features and the target")
    save_fig(fig, "correlation_heatmap.png")
    charts.append("correlation_heatmap.png")

summary = {
    "task_type": TASK,
    "target": TARGET,
    "n_rows": int(len(df)),
    "n_features": int(X.shape[1]),
    "n_numeric_tested": sum(1 for r in records if r["kind"] == "numeric"),
    "n_categorical_tested": sum(1 for r in records if r["kind"] == "categorical"),
    "n_significant": sum(1 for r in records if r["significant"]),
    "skipped_features": skipped,
    "top_drivers": top_drivers,
    "all_tests": records,
    "multicollinearity": multicollinear[:10],
    "charts": charts,
}
if TASK == "classification":
    summary.update(
        positive_label=str(PARAMS.get("positive_label")),
        base_rate=base_rate,
        n_positive=int(is_pos.sum()),
    )
else:
    summary.update(target_mean=float(y.mean()), target_median=float(y.median()), target_std=float(y.std()))
save_json(summary, "stats.json")
log(f"tested {len(records)} features, {summary['n_significant']} significant, {len(charts)} charts")
