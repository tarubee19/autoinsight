"""Output contracts: what each code-writing agent must have produced.

LLM code can run without errors and still produce nonsense (wrong target encoding, leftover NaNs,
hallucinated feature names). These validators turn such silent failures into explicit feedback
that is fed back to the agent for its next attempt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

MIN_ROWS = 30


def _read_json(path: Path, problems: list[str], what: str) -> dict | None:
    if not path.is_file():
        problems.append(f"{what} was not written (call save_json(..., '{path.name}'))")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"{what} is not valid JSON: {exc}")
        return None
    if not isinstance(data, dict):
        problems.append(f"{what} must be a JSON object")
        return None
    return data


def validate_engineer(out_dir: Path, plan: dict, raw_rows: int) -> list[str]:
    problems: list[str] = []
    csv_path = out_dir / "cleaned.csv"
    if not csv_path.is_file():
        problems.append("cleaned.csv was not written (call save_csv(df, 'cleaned.csv'))")
    else:
        try:
            df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
        except Exception as exc:  # noqa: BLE001
            problems.append(f"cleaned.csv cannot be read: {exc}")
            df = None
        if df is not None:
            target = plan["target"]
            if target not in df.columns:
                problems.append(f"target column '{target}' is missing from cleaned.csv (columns: {list(df.columns)[:20]})")
            else:
                if df[target].isna().any():
                    problems.append(f"target '{target}' still contains {int(df[target].isna().sum())} missing values - drop those rows")
                elif plan["task_type"] == "classification":
                    values = set(df[target].unique())
                    if not values <= {0, 1} or len(values) != 2:
                        problems.append(f"classification target must be integer 0/1 with both classes present; found values {sorted(map(str, values))[:6]}")
                elif not pd.api.types.is_numeric_dtype(df[target]):
                    problems.append(f"regression target '{target}' must be numeric")
            if len(df) < MIN_ROWS:
                problems.append(f"only {len(df)} rows remain (need >= {MIN_ROWS}); you removed too much data")
            elif raw_rows and len(df) < 0.5 * raw_rows:
                problems.append(f"cleaning removed more than half of the rows ({raw_rows} -> {len(df)}); impute instead of dropping")
            if df.shape[1] < 2:
                problems.append("no feature columns remain besides the target")
            features = [c for c in df.columns if c != target]
            leftover = [c for c in features if df[c].isna().any()]
            if leftover:
                problems.append(f"missing values remain after imputation in: {leftover[:8]}")
            ids = [c for c in plan.get("id_columns", []) + plan.get("drop_columns", []) if c in features]
            if ids:
                problems.append(f"columns that must be dropped are still present: {ids[:8]}")
    log = _read_json(out_dir / "cleaning_log.json", problems, "cleaning_log.json")
    if log is not None:
        missing = [k for k in ("rows_raw", "rows_clean", "steps") if k not in log]
        if missing:
            problems.append(f"cleaning_log.json is missing keys: {missing}")
    return problems


def validate_analyst(out_dir: Path, plan: dict, cleaned_csv: Path) -> list[str]:
    problems: list[str] = []
    stats = _read_json(out_dir / "stats.json", problems, "stats.json")
    if stats is not None:
        drivers = stats.get("top_drivers")
        if not isinstance(drivers, list) or not drivers:
            problems.append("stats.json['top_drivers'] must be a non-empty list")
        else:
            columns = set(pd.read_csv(cleaned_csv, nrows=5, keep_default_na=False, na_values=[""]).columns)
            for i, d in enumerate(drivers[:10]):
                if not isinstance(d, dict) or not {"feature", "strength", "finding"} <= set(d):
                    problems.append(f"top_drivers[{i}] needs the keys feature, strength, finding")
                    break
                if d["feature"] not in columns:
                    problems.append(f"top_drivers[{i}] names feature '{d['feature']}' which is not a column of the cleaned data")
                    break
                if not isinstance(d["strength"], (int, float)) or not 0 <= d["strength"] <= 1:
                    problems.append(f"top_drivers[{i}]['strength'] must be a number between 0 and 1")
                    break
        if plan["task_type"] == "classification" and not isinstance(stats.get("base_rate"), (int, float)):
            problems.append("stats.json needs a numeric 'base_rate' (share of rows with the target event)")
    charts = [p for p in (out_dir / "charts").glob("*.png") if p.stat().st_size > 2000] if (out_dir / "charts").is_dir() else []
    if len(charts) < 2:
        problems.append(f"at least 2 non-empty charts are required via save_fig; found {len(charts)}")
    return problems


def normalise_stats(out_dir: Path) -> None:
    """Fill optional fields so the reporter can rely on one shape, whoever wrote stats.json."""
    path = out_dir / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    for d in stats.get("top_drivers", []):
        strength = float(d.get("strength", 0.0))
        d["strength"] = strength
        d.setdefault("kind", "numeric")
        d.setdefault(
            "strength_label",
            "negligible" if strength < 0.1 else "small" if strength < 0.3 else "medium" if strength < 0.5 else "large",
        )
        p_adj = d.get("p_adjusted")
        d.setdefault("significant", bool(p_adj is not None and p_adj < 0.05))
        d.setdefault("p_adjusted", p_adj)
    stats.setdefault("all_tests", stats["top_drivers"])
    stats["charts"] = sorted(p.name for p in (out_dir / "charts").glob("*.png"))
    # Replace the file instead of opening it for writing: in Docker mode it is owned by the sandbox uid, so the
    # host user may not write to it, but it can always replace a file inside the (world-writable) output dir.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    os.replace(tmp, path)
