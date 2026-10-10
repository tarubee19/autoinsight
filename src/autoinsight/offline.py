"""The 'offline' brain: answers agent requests without any network or API key.

* planner  -> the heuristic plan computed from the data profile
* engineer -> the vetted cleaning program  (autoinsight/scripts/engineer_template.py)
* analyst  -> the vetted statistics program (autoinsight/scripts/analyst_template.py)
* reporter -> :func:`write_narrative`, a deterministic writer that turns the structured facts into
  the six report sections. It is also the safety net used when an LLM report fails the number
  grounding check.

Everything it states comes from the facts dict, so the output is auditable by construction.
"""

from __future__ import annotations

import json

from . import templates
from .llm import LLMRequest


def respond(req: LLMRequest) -> str:
    if req.role == "planner":
        return json.dumps(req.context["heuristic_plan"])
    if req.role == "engineer":
        return "```python\n" + templates.source(templates.ENGINEER) + "```"
    if req.role == "analyst":
        return "```python\n" + templates.source(templates.ANALYST) + "```"
    if req.role == "reporter":
        return write_narrative(req.context["facts"])
    raise ValueError(f"offline client has no answer for role {req.role!r}")


# ---------------------------------------------------------------------------------------------
# Deterministic report writer
# ---------------------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _sentence(text: str) -> str:
    return str(text).strip().rstrip(".") + "."


def _human_list(items: list[str], limit: int = 4) -> str:
    items = [str(i) for i in items]
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" and {len(items) - limit} more"


def write_narrative(facts: dict) -> str:
    ds, dq, model = facts["dataset"], facts["data_quality"], facts.get("model")
    drivers = facts["drivers"]
    classification = ds["task_type"] == "classification"
    target = ds["target"]
    sig = [d for d in drivers if d["significant"]]
    lines: list[str] = []

    # ---- Executive summary ------------------------------------------------------------------
    lines.append("## Executive summary")
    if classification:
        lines.append(
            f"- Across {ds['rows_clean']:,} analysed rows, {_pct(ds['base_rate'])} have **{target} = {ds['positive_label']}**."
        )
    else:
        lines.append(f"- Across {ds['rows_clean']:,} analysed rows, the median **{target}** is {ds['target_median']:,.2f}.")
    if sig:
        top = sig[:3]
        lines.append(
            "- The strongest statistically significant drivers are "
            + ", ".join(f"**{d['feature']}** ({d['strength_label']} association, strength {d['strength']:.2f})" for d in top)
            + "."
        )
        lines.append(f"- {_sentence(top[0]['finding'])}")
    else:
        lines.append("- No feature shows a statistically significant association after multiple-testing correction.")
    if model:
        if classification:
            lines.append(
                f"- A {model['model']} model separates the outcome on unseen data with ROC AUC {model['roc_auc']:.2f}; "
                f"the highest-scored 10% of rows contain {model['capture_top_decile']:.0%} of all positives "
                f"({model['lift_top_decile']:.1f}x lift over random selection)."
            )
        else:
            lines.append(
                f"- A {model['model']} model explains {model['r2']:.0%} of the variance on unseen data "
                f"(R2 {model['r2']:.2f}, typical error {model['mae']:,.2f})."
            )
    else:
        lines.append("- Predictive modelling was not possible for this dataset (see limitations).")
    lines.append("- These are statistical associations in observational data, not proof of cause and effect.")

    # ---- Data quality ------------------------------------------------------------------------
    lines.append("\n## Data quality & preparation")
    lines.append(
        f"- Started with {dq['rows_raw']:,} rows and {dq['cols_raw']} columns; kept {dq['rows_clean']:,} rows and {dq['cols_clean']} columns after cleaning."
    )
    if dq["duplicates_removed"]:
        lines.append(f"- Removed {dq['duplicates_removed']:,} exact duplicate rows so no record is counted twice.")
    if dq["dropped_columns"]:
        dropped = [f"{c} ({r})" for c, r in dq["dropped_columns"].items()]
        lines.append(f"- Dropped {len(dropped)} columns that cannot carry signal: {_human_list(dropped, 5)}.")
    if dq["coerced_to_numeric"]:
        lines.append(f"- Converted text columns that actually hold numbers: {_human_list(list(dq['coerced_to_numeric']))}.")
    if dq["imputed"]:
        worst = sorted(dq["imputed"].items(), key=lambda kv: -kv[1]["share"])[:3]
        lines.append(
            "- Filled missing values in "
            f"{len(dq['imputed'])} columns using the median (numbers) or most common value (categories); most affected: "
            + ", ".join(f"{c} ({v['share']:.2%} missing)" for c, v in worst)
            + "."
        )
    if dq["clipped_outliers"]:
        lines.append(
            f"- Clipped extreme values in {_human_list(list(dq['clipped_outliers']))} to the 0.5-99.5 percentile range so entry errors cannot distort the results."
        )
    if not (dq["duplicates_removed"] or dq["dropped_columns"] or dq["imputed"] or dq["clipped_outliers"]):
        lines.append("- The data needed little cleaning.")

    # ---- Drivers -----------------------------------------------------------------------------
    lines.append("\n## What drives the outcome")
    if drivers:
        lines.append(
            "Features are ranked by association strength (0 = none, 1 = perfect); significance is after Benjamini-Hochberg correction for testing many features."
        )
        for rank, d in enumerate(drivers[:6], 1):
            flag = "significant" if d["significant"] else "not significant"
            lines.append(f"{rank}. **{d['feature']}** - strength {d['strength']:.2f} ({d['strength_label']}, {flag}). {_sentence(d['finding'])}")
    for warn in facts.get("multicollinearity", [])[:2]:
        lines.append(
            f"- Note: **{warn['a']}** and **{warn['b']}** move together (rank correlation {warn['spearman_r']:.2f}); their individual effects overlap."
        )

    # ---- Model -------------------------------------------------------------------------------
    lines.append("\n## Predictive model")
    if model:
        comp = ", ".join(f"{c['model']} {c['cv_mean']:.2f}" for c in model["comparison"])
        metric_name = "ROC AUC" if classification else "R2"
        lines.append(
            f"- Three candidates were compared with {model['cv_folds']}-fold cross-validation on the training split ({metric_name}): {comp}. The best, **{model['model']}**, was evaluated once on a held-out 20% test set."
        )
        if classification:
            lines.append(
                f"- Test performance: ROC AUC {model['roc_auc']:.2f}, precision {model['precision']:.2f}, recall {model['recall']:.2f}, "
                f"F1 {model['f1']:.2f} at a decision threshold of {model['threshold']:.2f} (chosen on training data only)."
            )
            lines.append(
                f"- Always predicting the majority class would be right {model['baseline_accuracy']:.0%} of the time but would find none of the positives; the model's recall of {model['recall']:.2f} is what makes it useful."
            )
        else:
            lines.append(f"- Test performance: R2 {model['r2']:.2f}, MAE {model['mae']:,.2f}, RMSE {model['rmse']:,.2f}.")
        imp = facts.get("model_importance", [])[:5]
        if imp:
            lines.append("- What the model relies on most (permutation importance on held-out data): " + _human_list([i["feature"] for i in imp], 5) + ".")
        for w in model.get("warnings", []):
            lines.append(f"- Warning: {w}")
    else:
        lines.append("- No model was trained: " + facts.get("model_error", "not enough usable data") + ".")

    # ---- Recommendations ---------------------------------------------------------------------
    lines.append("\n## Recommendations")
    cat = [d for d in sig if d["kind"] == "categorical" and d.get("top_level") is not None][:2]
    num = [d for d in sig if d["kind"] == "numeric"][:2]
    n = 1
    if classification and cat:
        segs = "; ".join(f"{d['feature']} = '{d['top_level']}' ({_pct(d['top_rate'])} vs {_pct(ds['base_rate'])} overall)" for d in cat)
        lines.append(f"{n}. Prioritise the highest-rate segments first: {segs}.")
        n += 1
    if num:
        lines.append(f"{n}. Investigate why **{num[0]['feature']}** separates the groups (it is {num[0].get('direction', 'different')} for the positive class) and whether it can be influenced.")
        n += 1
    if model and classification:
        lines.append(f"{n}. Use the model's risk score to rank cases and act on the top decile first, where it captures {model['capture_top_decile']:.0%} of positives.")
        n += 1
    elif model:
        lines.append(f"{n}. Use the model for estimates, and focus data-quality work on the features it relies on most.")
        n += 1
    lines.append(f"{n}. Validate the top drivers with a controlled experiment before committing budget - these results show association, not causation.")
    n += 1
    gaps = [c for c, v in dq["imputed"].items() if v["share"] >= 0.05]
    if gaps:
        lines.append(f"{n}. Improve data capture for {_human_list(gaps)}; more than 5% of values were missing.")

    # ---- Limitations -------------------------------------------------------------------------
    lines.append("\n## Limitations & next steps")
    if ds.get("target_selected_by") != "user":
        lines.append(f"- The target column **{target}** was inferred from your question; if that is not what you meant, run again with the target column set explicitly.")
    lines.append("- Observational data: drivers are associations, and hidden confounders may explain part of them.")
    lines.append("- Models were validated on a single random split; for time-dependent data use a time-based split before deploying.")
    lines.append("- Missing values were imputed and extreme values clipped; both are logged in Appendix A and can change effect sizes slightly.")
    if facts.get("skipped_features"):
        lines.append(f"- Skipped from testing: {_human_list([s['feature'] for s in facts['skipped_features']])}.")
    lines.append("- Next steps: add domain features, test a time-based validation, and monitor drift after deployment.")
    return "\n".join(lines)
