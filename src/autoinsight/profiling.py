"""Dataset loading, profiling and heuristic planning.

The *profile* is the only view of the data that LLM prompts ever contain: column names, types,
missing-value rates and (optionally) 3 example values per column. Raw rows are never sent.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

POSITIVE_TOKENS = {
    "yes", "y", "true", "t", "1", "churn", "churned", "left", "attrited", "attrition",
    "positive", "fraud", "default", "defaulted", "bad", "cancelled", "canceled",
}  # fmt: skip
TARGET_NAME_HINTS = (
    "churn", "attrition", "default", "fraud", "target", "label", "outcome", "converted",
    "conversion", "cancel", "response", "survived", "readmitted", "click",
)  # fmt: skip
# Outcome words people use in questions -> the column-name roots they usually mean ("why do employees
# leave?" -> Attrition). Pure token overlap cannot bridge that gap; the LLM planner can, this is the
# no-LLM approximation.
OUTCOME_SYNONYMS = {
    "churn": {"leave", "left", "leaving", "quit", "cancel", "cancelled", "canceled", "retain", "retention", "defect", "lapse", "lost"},
    "attrition": {"leave", "left", "leaving", "quit", "resign", "resignation", "turnover", "retain", "retention"},
    "default": {"repay", "delinquent", "delinquency", "defaulted"},
    "fraud": {"fraudulent", "scam", "suspicious"},
    "convert": {"buy", "purchase", "signup", "subscribe"},
    "survived": {"survive", "survival", "die", "death", "mortality"},
    "readmitted": {"readmit", "readmission"},
    "click": {"ctr"},
    "cancel": {"cancellation"},
}  # fmt: skip
ID_NAME = re.compile(r"(^|[_\s-])(id|uuid|guid|key|number|no|code|index)$|^id([_\s-]|$)|identifier", re.I)
ID_CAMEL = re.compile(r"(?<=[a-z0-9])(Id|ID|Uuid|Guid|Key|Number|Code|Index)$")  # EmployeeNumber, customerID
DATE_VALUE = re.compile(r"^\s*(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")


MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "?", "#n/a", "missing", "--"}  # NOT "none": often a real category


class PlanError(ValueError):
    """The request cannot be turned into a valid analysis plan (message is user-facing)."""


def normalise_label(value) -> str:
    """Canonical string form of a class label so 1, 1.0, True and ' 1 ' compare equal."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip().lower()


def load_dataframe(path: str | Path, normalise_missing: bool = True) -> pd.DataFrame:
    """Read a CSV robustly: encoding fallback, delimiter sniffing, stripped headers.

    ``normalise_missing=False`` keeps placeholder tokens ("N/A", "?") as text, used when canonicalising
    the upload so that the cleaning agent - and its audit log - still sees the original values.
    """
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            opts = {"encoding": encoding, "keep_default_na": False, "na_values": [""]}
            df = pd.read_csv(path, low_memory=False, **opts)
            if df.shape[1] == 1:  # probably a ; or \t separated file
                df = pd.read_csv(path, sep=None, engine="python", **opts)
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    else:
        raise PlanError(f"Could not read the file as CSV: {last_error}")
    if df.empty:
        raise PlanError("The CSV file has no rows.")
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns if normalise_missing else []:
        if not (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])):
            is_token = df[col].map(lambda v: isinstance(v, str) and v.strip().lower() in MISSING_TOKENS)
            if is_token.any():
                df[col] = df[col].where(~is_token, np.nan)
    if df.columns.duplicated().any():
        raise PlanError("The CSV has duplicate column names; please rename them.")
    return df


def _kind_of(name: str, series: pd.Series, n_rows: int) -> str:
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        is_int = pd.api.types.is_integer_dtype(series) or bool(np.allclose(non_null, non_null.round())) if len(non_null) else False
        if is_int and n_unique >= 0.98 * max(len(non_null), 1) and (ID_NAME.search(name) or ID_CAMEL.search(name)):
            return "id"
        return "numeric"
    sample = non_null.astype(str).str.strip()
    if sample.empty:
        return "empty"
    cleaned = sample.str.replace(r"[,$%\s]", "", regex=True)
    if pd.to_numeric(cleaned, errors="coerce").notna().mean() >= 0.9 and n_unique > 10:
        return "numeric_text"
    if sample.head(200).str.match(DATE_VALUE).mean() >= 0.9:
        return "datetime_text"
    if n_unique >= 0.95 * len(non_null) and (ID_NAME.search(name) or ID_CAMEL.search(name) or n_unique > 50):
        return "id"
    if n_unique > max(50, 0.5 * n_rows):
        return "text"
    return "categorical"


def profile_dataframe(df: pd.DataFrame, send_samples: bool = True, max_columns: int = 120) -> dict:
    n_rows = len(df)
    columns = []
    for name in list(df.columns)[:max_columns]:
        series = df[name]
        entry: dict = {
            "name": name,
            "dtype": str(series.dtype),
            "kind": _kind_of(name, series, n_rows),
            "n_missing": int(series.isna().sum()),
            "pct_missing": round(float(series.isna().mean()), 4),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if entry["kind"] == "numeric":
            desc = series.describe()
            entry.update({k: round(float(desc[k]), 4) for k in ("mean", "std", "min", "50%", "max") if k in desc})
        if entry["kind"] in {"categorical", "boolean"} and entry["n_unique"] <= 12:
            counts = series.value_counts(dropna=True).head(12)
            entry["value_counts"] = {str(k): int(v) for k, v in counts.items()}
        elif send_samples:
            entry["examples"] = [str(v) for v in series.dropna().astype(str).head(3)]
        columns.append(entry)
    return {
        "n_rows": n_rows,
        "n_cols": int(df.shape[1]),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
        "columns_not_shown": max(0, df.shape[1] - max_columns),
    }


STOP_WORDS = {
    "the", "and", "for", "what", "which", "why", "how", "find", "are", "with", "from", "that",
    "this", "does", "did", "can", "you", "our", "driving", "drivers", "driver", "factors",
    "factor", "behind", "predict", "analyze", "analyse", "analysis", "impact", "affect",
    "affects", "influence", "reasons", "reason", "main", "key", "top", "most", "explain",
}  # fmt: skip


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) >= 3 and t not in STOP_WORDS}


def detect_target(df: pd.DataFrame, question: str, hint: str | None = None) -> str:
    """Pick the target column: explicit hint > column named in the question > well-known names."""
    by_lower = {c.lower(): c for c in df.columns}
    if hint:
        key = hint.strip().lower()
        if key in by_lower:
            return by_lower[key]
        raise PlanError(f"Target column '{hint}' not found. Columns: {', '.join(map(str, df.columns))}")

    # Identifiers and free text can never be the thing we explain, even if their name matches.
    eligible = [
        c for c in df.columns
        if df[c].nunique(dropna=True) >= 2 and _kind_of(c, df[c], len(df)) not in {"id", "text", "empty"}
    ]  # fmt: skip
    q_tokens = _tokens(question)
    scored = []
    for col in eligible:
        col_flat = re.sub(r"[^a-z0-9]", "", col.lower())
        col_tokens = _tokens(re.sub(r"([a-z])([A-Z])", r"\1 \2", col))
        overlap = col_tokens & q_tokens
        # stem-ish match: "churn" ~ "churned", "employees" ~ "employee"
        overlap |= {t for t in col_tokens for q in q_tokens if len(t) >= 4 and (q.startswith(t) or t.startswith(q))}
        synonym_hit = any(root in col_flat and (words & q_tokens) for root, words in OUTCOME_SYNONYMS.items())
        is_hint = any(h in col.lower() for h in TARGET_NAME_HINTS)
        score = 2.0 * len(overlap) + (3.0 if synonym_hit else 0.0)
        if score or is_hint:
            score += (1.0 if df[col].nunique(dropna=True) == 2 else 0.0) + (0.5 if is_hint else 0.0)
        if score >= 1.5:  # real overlap/synonym evidence, or a binary column with an outcome-like name
            scored.append((score, -len(str(col)), col))
    if scored:
        return max(scored)[2]
    for col in eligible:
        if any(h in col.lower() for h in TARGET_NAME_HINTS):
            return col
    raise PlanError(
        "Could not tell which column to predict/explain. Mention it in your question (e.g. "
        f"'drivers of <column>') or pass it explicitly. Columns: {', '.join(map(str, df.columns))}"
    )


def choose_positive_label(values: list) -> tuple[str, str]:
    """Return (positive, negative) labels for a binary target."""
    labels = [str(v).strip() for v in values]
    keyed = {normalise_label(v): str(v).strip() for v in values}
    hits = [k for k in keyed if k in POSITIVE_TOKENS]
    if len(hits) == 1:
        pos = keyed[hits[0]]
        neg = next(v for k, v in keyed.items() if k != hits[0])
        return pos, neg
    return labels[0], labels[1]  # caller re-orders by minority class


def build_heuristic_plan(df: pd.DataFrame, profile: dict, question: str, hint: str | None = None) -> dict:
    target = detect_target(df, question, hint)
    series = df[target].dropna()
    n_unique = series.nunique()
    if n_unique < 2:
        raise PlanError(f"Target '{target}' has a single value - nothing to explain.")

    plan: dict = {"target": target, "source": "heuristic"}
    if n_unique == 2:
        plan["task_type"] = "classification"
        counts = series.map(normalise_label).value_counts()
        raw_by_key = {normalise_label(v): str(v).strip() for v in series.unique()}
        keys = list(counts.index)
        hits = [k for k in keys if k in POSITIVE_TOKENS]
        pos_key = hits[0] if len(hits) == 1 else keys[-1]  # minority class if no signal
        neg_key = next(k for k in keys if k != pos_key)
        plan["positive_label"], plan["negative_label"] = raw_by_key[pos_key], raw_by_key[neg_key]
    elif pd.api.types.is_numeric_dtype(series) and n_unique >= 3:
        plan["task_type"] = "regression"
        plan["positive_label"] = plan["negative_label"] = None
    else:
        raise PlanError(
            f"Target '{target}' has {n_unique} classes. This version supports binary classification "
            "(2 classes) and regression (numeric target)."
        )

    t_norm = re.sub(r"[^a-z0-9]", "", target.lower())
    leakage = [
        c for c in df.columns
        if c != target and len(t_norm) >= 4 and t_norm in re.sub(r"[^a-z0-9]", "", c.lower())
    ]  # fmt: skip
    ids = [c["name"] for c in profile["columns"] if c["kind"] in {"id", "empty", "text"} and c["name"] != target]
    plan.update(
        id_columns=ids,
        leakage_suspects=leakage,
        drop_columns=leakage,
        hypotheses=[],
        rationale=f"Target '{target}' selected from the question/column names; task: {plan['task_type']}.",
    )
    return plan


def normalise_plan(raw: dict, df: pd.DataFrame, profile: dict, question: str, hint: str | None) -> dict:
    """Validate an LLM-produced plan against the real data; fill gaps from the heuristic plan.

    Raises :class:`PlanError` if the LLM named a target that does not exist (the caller then
    retries once and finally falls back to the heuristic plan).
    """
    cols = {c.lower(): c for c in df.columns}
    chosen = hint or str(raw.get("target", "")).strip()
    if chosen.lower() not in cols:
        raise PlanError(f"Planner chose a target that does not exist: {chosen!r}")
    plan = build_heuristic_plan(df, profile, question, cols[chosen.lower()])
    target = plan["target"]

    def valid(key: str) -> set[str]:
        return {cols[str(v).lower()] for v in (raw.get(key) or []) if str(v).lower() in cols} - {target}

    plan["id_columns"] = sorted(set(plan["id_columns"]) | valid("id_columns"))
    plan["leakage_suspects"] = sorted(set(plan["leakage_suspects"]) | valid("leakage_suspects"))
    plan["drop_columns"] = sorted(set(plan["leakage_suspects"]) | valid("drop_columns"))

    if plan["task_type"] == "classification":
        values = {normalise_label(v): str(v).strip() for v in df[target].dropna().unique()}
        pos = raw.get("positive_label")
        if pos is not None and normalise_label(pos) in values:
            plan["positive_label"] = values[normalise_label(pos)]
            plan["negative_label"] = next(v for k, v in values.items() if k != normalise_label(pos))
    plan["hypotheses"] = [str(h)[:200] for h in (raw.get("hypotheses") or [])][:5]
    plan["rationale"] = str(raw.get("rationale") or plan["rationale"])[:500]
    plan["source"] = "llm"
    return plan
