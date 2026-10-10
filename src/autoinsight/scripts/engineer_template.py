# Vetted baseline for the Data Engineer agent.
#
# This program obeys the exact same contract as LLM-written code (runs in the sandbox, passes the
# AST guard) and is used (a) by the offline provider and (b) as the deterministic fallback when an
# LLM cannot produce working code within the allowed number of attempts.
#
# Contract
#   inputs : INPUT_PATH (raw CSV), PARAMS (target, task_type, positive_label, drop_columns, ...)
#   outputs: cleaned.csv, cleaning_log.json

import numpy as np
import pandas as pd

TARGET = PARAMS["target"]
TASK = PARAMS["task_type"]
POSITIVE = PARAMS.get("positive_label")
PLAN_DROP = [c for c in PARAMS.get("drop_columns", []) + PARAMS.get("id_columns", []) if c != TARGET]
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "?", "#n/a", "missing", "--"}

steps = []
dropped = {}
imputed = {}
clipped = {}
coerced = {}
date_features = {}


def add_step(name, detail, count=None):
    steps.append({"step": name, "detail": detail, "count": count})


def is_text(series):
    return not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series))


df = pd.read_csv(INPUT_PATH, low_memory=False, keep_default_na=False, na_values=[""])
rows_raw, cols_raw = df.shape
df.columns = [str(c).strip() for c in df.columns]
if TARGET not in df.columns:
    raise ValueError(f"target column {TARGET!r} not found; columns are {list(df.columns)}")

# 1. Trim whitespace and turn placeholder tokens ("", "N/A", "?", ...) into real missing values
n_tokens = 0
for col in [c for c in df.columns if is_text(df[c])]:
    trimmed = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    is_token = trimmed.map(lambda v: isinstance(v, str) and v.lower() in MISSING_TOKENS)
    n_tokens += int(is_token.sum())
    df[col] = trimmed.where(~is_token, np.nan)
if n_tokens:
    add_step("placeholder_to_missing", "Converted blank / 'N/A' / '?' style placeholders to missing values", n_tokens)

# 2. Exact duplicate rows
n_dup = int(df.duplicated().sum())
if n_dup:
    df = df.drop_duplicates().reset_index(drop=True)
    add_step("drop_duplicates", "Removed exact duplicate rows", n_dup)

# 3. Numbers stored as text ("1,234", "$45", " 29.9 ") -> numeric
for col in [c for c in df.columns if c != TARGET and is_text(df[c])]:
    non_null = df[col].dropna()
    if non_null.empty:
        continue
    cleaned = non_null.astype("string").str.replace(r"[,$%\s]", "", regex=True)
    converted = pd.to_numeric(cleaned, errors="coerce")
    ok_share = float(converted.notna().mean())
    if ok_share >= 0.9:
        full = pd.to_numeric(df[col].astype("string").str.replace(r"[,$%\s]", "", regex=True), errors="coerce")
        lost = int(df[col].notna().sum() - full.notna().sum())
        df[col] = full.astype("float64")  # plain numpy float, not a nullable extension dtype
        coerced[col] = {"non_numeric_entries_set_to_missing": lost}
if coerced:
    add_step("coerce_numeric", "Converted text columns that are really numbers", len(coerced))

# 4. Date-like text columns -> year / month / weekday features
for col in [c for c in df.columns if c != TARGET and is_text(df[c])]:
    sample = df[col].dropna().astype(str).head(200)
    if sample.empty:
        continue
    looks_like_date = sample.str.match(r"^\d{4}-\d{1,2}-\d{1,2}|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}").mean() >= 0.9
    if not looks_like_date:
        continue
    parsed = pd.to_datetime(df[col], errors="coerce")
    if parsed.notna().mean() < 0.9:
        continue
    df[col + "_year"] = parsed.dt.year
    df[col + "_month"] = parsed.dt.month
    df[col + "_dow"] = parsed.dt.dayofweek
    df = df.drop(columns=[col])
    date_features[col] = [col + "_year", col + "_month", col + "_dow"]
if date_features:
    add_step("expand_dates", "Replaced date columns by year / month / weekday", len(date_features))

# 5. Drop columns that cannot help: ids, constants, mostly-missing, free text
for col in PLAN_DROP:
    if col in df.columns:
        dropped[col] = "identifier / excluded by the plan"
n_rows = len(df)
for col in [c for c in df.columns if c != TARGET and c not in dropped]:
    series = df[col]
    n_unique = series.nunique(dropna=True)
    if n_unique <= 1:
        dropped[col] = "constant column"
    elif series.isna().mean() > 0.6:
        dropped[col] = f"{series.isna().mean():.0%} missing"
    elif is_text(series) and n_unique > max(50, 0.5 * n_rows):
        dropped[col] = f"high-cardinality text ({n_unique} distinct values)"
if dropped:
    df = df.drop(columns=list(dropped))
    add_step("drop_columns", "Dropped columns that carry no usable signal", len(dropped))


# 6. Target: drop rows with no label, encode classification target as 0/1
def normalise(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip().lower()


n_before = len(df)
df = df[df[TARGET].notna()].reset_index(drop=True)
if len(df) < n_before:
    add_step("drop_unlabelled_rows", "Removed rows whose target is missing", n_before - len(df))

target_info = {"task_type": TASK}
if TASK == "classification":
    keys = df[TARGET].map(normalise)
    levels = sorted(keys.unique())
    if len(levels) != 2:
        raise ValueError(f"classification target must have exactly 2 classes, found {levels[:10]}")
    positive_key = normalise(POSITIVE)
    if positive_key not in levels:
        raise ValueError(f"positive_label {POSITIVE!r} not among target values {levels}")
    df[TARGET] = (keys == positive_key).astype(int)
    target_info.update(
        positive_label=str(POSITIVE),
        negative_label=str([v for v in levels if v != positive_key][0]),
        positive_rate=float(df[TARGET].mean()),
        n_positive=int(df[TARGET].sum()),
    )
    add_step("encode_target", f"Encoded target as 1 = '{POSITIVE}', 0 = other", len(df))
else:
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype("float64")
    n_bad = int(df[TARGET].isna().sum())
    if n_bad:
        df = df[df[TARGET].notna()].reset_index(drop=True)
        add_step("drop_non_numeric_target", "Removed rows with non-numeric target", n_bad)
    target_info.update(mean=float(df[TARGET].mean()), std=float(df[TARGET].std()))

features = [c for c in df.columns if c != TARGET]
if not features:
    raise ValueError("no usable feature columns left after cleaning")
if len(df) < 30:
    raise ValueError(f"only {len(df)} rows left after cleaning - too few to analyse")

# 7. Winsorise extreme outliers of continuous features (0.5th / 99.5th percentile)
if len(df) >= 200:
    for col in features:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
            low, high = series.quantile(0.005), series.quantile(0.995)
            n_out = int(((series < low) | (series > high)).sum())
            if n_out:
                df[col] = series.clip(low, high)
                clipped[col] = n_out
if clipped:
    add_step("winsorise_outliers", "Clipped extreme values to the 0.5-99.5 percentile range", sum(clipped.values()))

# 8. Impute missing values (median / mode); add an indicator when >= 5% was missing
for col in features:
    n_missing = int(df[col].isna().sum())
    if n_missing == 0:
        continue
    share = n_missing / len(df)
    if pd.api.types.is_numeric_dtype(df[col]):
        if share >= 0.05:
            df[col + "__was_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())
        imputed[col] = {"strategy": "median", "n_missing": n_missing, "share": round(share, 4)}
    else:
        mode = df[col].mode(dropna=True)
        fill = "Missing" if (share >= 0.05 or mode.empty) else mode.iloc[0]
        df[col] = df[col].fillna(fill)
        imputed[col] = {"strategy": "constant 'Missing'" if fill == "Missing" else "mode", "n_missing": n_missing, "share": round(share, 4)}
if imputed:
    add_step("impute_missing", "Filled remaining missing values", sum(v["n_missing"] for v in imputed.values()))

save_csv(df, "cleaned.csv")
save_json(
    {
        "rows_raw": rows_raw,
        "rows_clean": len(df),
        "cols_raw": cols_raw,
        "cols_clean": df.shape[1],
        "duplicates_removed": n_dup,
        "placeholders_to_missing": n_tokens,
        "dropped_columns": dropped,
        "coerced_to_numeric": coerced,
        "date_features": date_features,
        "imputed": imputed,
        "clipped_outliers": clipped,
        "target": target_info,
        "steps": steps,
    },
    "cleaning_log.json",
)
log(f"cleaned {rows_raw} -> {len(df)} rows, {cols_raw} -> {df.shape[1]} columns")
