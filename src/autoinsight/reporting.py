"""Facts -> report. Builds the structured facts every agent output is reduced to, and assembles the
final Markdown/HTML report (narrative + charts + audit appendices)."""

from __future__ import annotations

import base64
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import markdown

from . import __version__

CHART_TITLES = {
    "target_distribution.png": "Target distribution",
    "driver_strength.png": "Strength of association with the target",
    "numeric_drivers.png": "Top numeric drivers",
    "categorical_drivers.png": "Top categorical drivers",
    "correlation_heatmap.png": "Correlation between top numeric features",
    "model_feature_importance.png": "What the model relies on",
    "roc_curve.png": "Model discrimination (ROC curve)",
    "pred_vs_actual.png": "Predicted vs actual",
}


def build_facts(
    *, question: str, plan: dict, cleaning: dict, stats: dict, metrics: dict | None,
    importance: dict | None, model_error: str | None = None,
) -> dict:  # fmt: skip
    """Reduce all artefacts to one compact JSON-able dict - the single source of truth for the
    report and for the number-grounding check."""
    classification = plan["task_type"] == "classification"
    by_feature = {t.get("feature"): t for t in stats.get("all_tests", [])}

    drivers = []
    for d in stats.get("top_drivers", [])[:8]:
        detail = by_feature.get(d["feature"], {})
        item = {k: d.get(k) for k in ("feature", "kind", "strength", "strength_label", "p_adjusted", "significant", "finding")}
        for key in ("top_level", "top_rate", "top_n", "low_level", "low_rate", "direction", "cohens_d",
                    "median_positive", "median_negative", "spearman_rho"):  # fmt: skip
            if key in detail:
                item[key] = detail[key]
        drivers.append(item)

    target_info = cleaning.get("target", {})
    dataset = {
        "rows_raw": cleaning.get("rows_raw", stats.get("n_rows")),
        "rows_clean": cleaning.get("rows_clean", stats.get("n_rows")),
        "cols_raw": cleaning.get("cols_raw"),
        "cols_clean": cleaning.get("cols_clean"),
        "target": plan["target"],
        "task_type": plan["task_type"],
        "positive_label": plan.get("positive_label"),
        "target_selected_by": plan.get("target_selected_by", "planner"),
    }
    if classification:
        dataset["base_rate"] = stats.get("base_rate", target_info.get("positive_rate"))
    else:
        dataset["target_median"] = stats.get("target_median")
        dataset["target_mean"] = stats.get("target_mean")

    facts = {
        "question": question,
        "dataset": dataset,
        "data_quality": {
            "rows_raw": dataset["rows_raw"], "rows_clean": dataset["rows_clean"],
            "cols_raw": dataset["cols_raw"], "cols_clean": dataset["cols_clean"],
            "duplicates_removed": cleaning.get("duplicates_removed", 0),
            "placeholders_to_missing": cleaning.get("placeholders_to_missing", 0),
            "dropped_columns": cleaning.get("dropped_columns", {}) or {},
            "coerced_to_numeric": cleaning.get("coerced_to_numeric", {}) or {},
            "imputed": _normalise_imputed(cleaning.get("imputed", {})),
            "clipped_outliers": cleaning.get("clipped_outliers", {}) or {},
        },  # fmt: skip
        "drivers": drivers,
        "multicollinearity": stats.get("multicollinearity", [])[:3],
        "skipped_features": stats.get("skipped_features", [])[:10],
        "n_features_tested": len(stats.get("all_tests", [])) or len(drivers),
        # Fixed methodology parameters, listed so the narrative can cite them and stay verifiable.
        "method": {
            "test_split_pct": 20, "top_slice_pct": 10, "outlier_clip_low_pct": 0.5,
            "outlier_clip_high_pct": 99.5, "significance_level": 0.05, "max_imputation_indicator_pct": 5,
        },
    }
    if metrics:
        keep = ("model", "cv_folds", "cv_metric", "n_train", "n_test", "threshold", "roc_auc", "pr_auc", "accuracy",
                "precision", "recall", "f1", "brier", "baseline_accuracy", "lift_top_decile",
                "capture_top_decile", "r2", "mae", "rmse", "warnings")  # fmt: skip
        model = {k: metrics[k] for k in keep if k in metrics}
        model["comparison"] = [
            {"model": c["model"], "cv_mean": c["cv_mean"], "cv_std": c["cv_std"]} for c in metrics.get("model_comparison", [])
        ]
        facts["model"] = model
        facts["model_importance"] = [
            {"feature": f["feature"], "importance": f["importance"], "share_of_total": f["share_of_total"],
             "direction": f.get("effect_direction") or f.get("highest_level")}  # fmt: skip
            for f in (importance or {}).get("features", [])[:8]
        ]
    else:
        facts["model_error"] = model_error or "model training was skipped"
    return facts


def _normalise_imputed(imputed: dict) -> dict:
    """LLM-written cleaning logs may describe imputation as a string; keep a uniform shape."""
    out = {}
    for col, info in (imputed or {}).items():
        if isinstance(info, dict):
            out[col] = {"strategy": info.get("strategy", "n/a"), "n_missing": info.get("n_missing", 0), "share": float(info.get("share", 0.0) or 0.0)}
        else:
            out[col] = {"strategy": str(info), "n_missing": 0, "share": 0.0}
    return out


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |")
    return "\n".join(out)


def assemble_report(*, narrative: str, facts: dict, cleaning: dict, stats: dict, metrics: dict | None,
                    chart_files: list[str], meta: dict) -> str:  # fmt: skip
    ds = facts["dataset"]
    header = [
        "# Automated Data Analysis Report",
        "",
        f"> **Question:** {facts['question']}  ",
        f"> **Target:** `{ds['target']}` ({ds['task_type']}) &nbsp;|&nbsp; **Rows analysed:** {ds['rows_clean']:,} "
        f"&nbsp;|&nbsp; **Generated:** {meta['generated_at']} &nbsp;|&nbsp; **Engine:** {meta['llm']}",
        "",
    ]
    charts = ["## Charts", ""]
    order = list(CHART_TITLES)
    for name in sorted(chart_files, key=lambda n: order.index(n) if n in order else len(order)):
        title = CHART_TITLES.get(name, name.replace("_", " ").replace(".png", "").capitalize())
        charts += [f"**{title}**", "", f"![{title}](charts/{name})", ""]

    steps = cleaning.get("steps", [])
    appendix_a = ["## Appendix A - Data cleaning log", ""]
    if steps:
        appendix_a.append(_table(["Step", "What was done", "Rows/cells/columns affected"], [[s.get("step"), s.get("detail"), s.get("count", "")] for s in steps]))
    else:
        appendix_a.append("_No cleaning steps were logged._")

    tests = sorted(stats.get("all_tests", []), key=lambda t: (not t.get("significant", False), -(t.get("strength") or 0)))[:15] or stats.get("top_drivers", [])
    appendix_b = ["", "## Appendix B - Statistical tests (top 15 by strength)", ""]
    appendix_b.append(_table(
        ["Feature", "Type", "Test", "Strength (0-1)", "Size", "Adj. p-value", "Significant"],
        [[t.get("feature"), t.get("kind"), t.get("test", ""), f"{(t.get('strength') or 0):.3f}", t.get("strength_label", ""),
          f"{t['p_adjusted']:.2g}" if t.get("p_adjusted") is not None else "n/a", "yes" if t.get("significant") else "no"] for t in tests],
    ))  # fmt: skip

    appendix_c = ["", "## Appendix C - Model evaluation", ""]
    if metrics:
        appendix_c.append(_table(["Candidate", f"CV {metrics.get('cv_metric', '')} (mean)", "CV std"],
                                 [[c["model"], f"{c['cv_mean']:.4f}", f"{c['cv_std']:.4f}"] for c in metrics.get("model_comparison", [])]))
        skip = {"model_comparison", "calibration_bins", "warnings", "task_type"}
        flat = [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in metrics.items() if k not in skip]
        appendix_c += ["", _table(["Held-out test metric", "Value"], flat)]
    else:
        appendix_c.append(f"_No model was trained: {facts.get('model_error', 'n/a')}_")

    g = meta.get("grounding", {})
    appendix_d = [
        "", "## Appendix D - Reproducibility & verification", "",
        _table(["Item", "Value"], [
            ["Run id", meta["run_id"]], ["AutoInsight version", __version__],
            ["LLM / engine", meta["llm"]], ["Code execution sandbox", meta["sandbox"]],
            ["Attempts (engineer / analyst)", f"{meta['attempts'].get('engineer', 0)} / {meta['attempts'].get('analyst', 0)}"],
            ["Deterministic fallback used", ", ".join(meta.get("fallbacks", [])) or "none"],
            ["Numbers in narrative traced to computed results", f"{g.get('numbers_grounded', 0)} of {g.get('numbers_checked', 0)} ({g.get('grounding_rate', 1):.0%})"],
            ["Random seed", meta["seed"]], ["Python", platform.python_version()],
        ]),
        "",
        "_Every figure in the narrative was checked against the computed results; the code each agent ran is stored in `logs/` next to this report._",
    ]  # fmt: skip
    return "\n".join(header) + "\n" + narrative.strip() + "\n\n" + "\n".join(charts + appendix_a + appendix_b + appendix_c + appendix_d) + "\n"


_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:920px;margin:2rem auto;padding:0 1.2rem;color:#0b0b0b;line-height:1.55}
h1{font-size:1.9rem;border-bottom:2px solid #2a78d6;padding-bottom:.3rem}h2{margin-top:2rem;border-bottom:1px solid #e6e5e1;padding-bottom:.2rem}
blockquote{background:#f5f7fa;border-left:4px solid #2a78d6;margin:1rem 0;padding:.6rem 1rem;color:#52514e}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.8rem 0}th,td{border:1px solid #e6e5e1;padding:.35rem .55rem;text-align:left}th{background:#f5f7fa}
img{max-width:100%;border:1px solid #e6e5e1;border-radius:6px;margin:.4rem 0 1rem}code{background:#f0efec;padding:.1rem .3rem;border-radius:3px}
@media (prefers-color-scheme: dark){body{background:#1a1a19;color:#f0efec}blockquote,th{background:#242423}code{background:#2c2c2a}table,th,td,img,h2{border-color:#3a3a38}}
"""


def to_html(md_text: str, base_dir: Path) -> str:
    """Self-contained HTML (charts inlined as base64) so the report can be e-mailed as one file."""
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])

    def inline(match: re.Match) -> str:
        src = match.group(1)
        path = (base_dir / src).resolve()
        if path.is_file() and base_dir.resolve() in path.parents:
            data = base64.b64encode(path.read_bytes()).decode()
            return f'src="data:image/png;base64,{data}"'
        return match.group(0)

    html_body = re.sub(r'src="([^"]+\.png)"', inline, html_body)
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Automated Data Analysis Report</title><style>{_CSS}</style></head><body>{html_body}</body></html>"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
