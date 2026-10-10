"""Prompt builders for the LLM-backed agents.

Prompts are plain functions (not a templating engine) so they are easy to read, test and tweak.
Each code-writing prompt states the *contract* the program must satisfy; the same contract is
enforced afterwards by the validators in ``agents/contracts.py``.
"""

from __future__ import annotations

import json

SANDBOX_RULES = """\
ENVIRONMENT (strict)
- Write ONE complete, self-contained Python 3 program. No input(), no CLI args.
- Already defined for you: INPUT_PATH (str, the CSV to read), PARAMS (dict, the analysis plan),
  save_csv(df, name), save_json(obj, name), save_fig(fig, name) (PNG, saved under charts/), log(msg), plt.
- Allowed imports ONLY: pandas, numpy, scipy, matplotlib, sklearn, statsmodels, seaborn, math, re,
  json, warnings, itertools, collections, statistics, datetime, typing, textwrap, random.
- FORBIDDEN (the program is rejected before it runs): os, sys, subprocess, socket, requests, pickle,
  open(), eval/exec, getattr/setattr, any dunder attribute, absolute paths, URLs, file writes other
  than save_csv/save_json/save_fig. There is no network.
- Read data ONLY with pd.read_csv(INPUT_PATH, keep_default_na=False, na_values=[""]) so that real
  categories such as "None" or "NA" are not silently turned into missing values.
  Write results ONLY with the save_* helpers.
- Be defensive: real data has missing values, numbers stored as text, constant columns, ids.
- Output only the code in one ```python block. No explanations."""


def _dump(obj, limit: int = 14000) -> str:
    text = json.dumps(obj, indent=1, default=str)
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


def planner_prompts(question: str, profile: dict, columns: list[str], hint: str | None) -> tuple[str, str]:
    system = (
        "You are the Planner agent of an automated data-analysis team. Given a business question and "
        "a data profile, decide what to analyse. Reply with ONE JSON object and nothing else."
    )
    user = f"""BUSINESS QUESTION
{question}

USER-SPECIFIED TARGET COLUMN: {hint or "(none)"}

AVAILABLE COLUMNS
{json.dumps(columns)}

DATA PROFILE
{_dump(profile)}

Return JSON with exactly these keys:
{{
  "target": "<exact column name to explain/predict>",
  "task_type": "classification" | "regression",
  "positive_label": "<value of the target that counts as the event, e.g. 'Yes'; null for regression>",
  "id_columns": ["columns that are identifiers"],
  "drop_columns": ["columns that must not be used (ids, leakage: derived from the outcome, or free text)"],
  "leakage_suspects": ["columns that look derived from the target"],
  "hypotheses": ["up to 4 short, testable hypotheses about drivers"],
  "rationale": "one sentence"
}}
Use only column names that exist in AVAILABLE COLUMNS."""
    return system, user


def _with_line_numbers(code: str) -> str:
    return "\n".join(f"{i:3d}| {line}" for i, line in enumerate(code.splitlines(), 1))


def _repair_block(previous_code: str | None, error: str | None) -> str:
    if not previous_code or not error:
        return ""
    return f"""

YOUR PREVIOUS PROGRAM FAILED
Previous program (with line numbers):
{_with_line_numbers(previous_code)}

Error / feedback:
{error}

Fix the root cause and return the COMPLETE corrected program (not a diff)."""


def engineer_prompts(question: str, profile: dict, plan: dict, previous_code: str | None, error: str | None) -> tuple[str, str]:
    system = (
        "You are the Data Engineer agent. You write robust pandas code that cleans a raw CSV so that "
        "downstream statistical analysis and modelling are reliable.\n\n" + SANDBOX_RULES
    )
    user = f"""BUSINESS QUESTION
{question}

PLAN (available as PARAMS in the program)
{_dump(plan, 4000)}

DATA PROFILE
{_dump(profile)}

REQUIRED OUTPUTS
1. save_csv(clean_df, "cleaned.csv")
   - Keep the target column under its original name PARAMS["target"].
   - Classification: target must be integer 0/1 where 1 means PARAMS["positive_label"].
     Regression: numeric target. Drop rows whose target is missing.
   - Drop identifier columns (PARAMS["id_columns"]), PARAMS["drop_columns"], constant columns,
     columns that are > 60% missing and high-cardinality free text.
   - Convert numbers stored as text (e.g. blanks, "1,234", "$45") to numeric. Treat "", "N/A", "?"
     as missing. Remove exact duplicate rows. Strip whitespace in text values.
   - Impute remaining missing values (numeric: median, categorical: mode or "Missing").
   - Clip extreme numeric outliers (0.5th-99.5th percentile) only for continuous features.
2. save_json({{...}}, "cleaning_log.json") with keys:
   rows_raw, rows_clean, cols_raw, cols_clean (ints), duplicates_removed (int),
   dropped_columns (dict column -> reason), imputed (dict column -> description),
   steps (list of dicts with keys "step", "detail", "count").
Log every change you make; never change data silently.{_repair_block(previous_code, error)}"""
    return system, user


def analyst_prompts(question: str, plan: dict, cleaned_profile: dict, previous_code: str | None, error: str | None) -> tuple[str, str]:
    system = (
        "You are the Analyst agent. You run rigorous statistical tests and make clear matplotlib "
        "charts that explain what drives the target.\n\n" + SANDBOX_RULES
    )
    user = f"""BUSINESS QUESTION
{question}

PLAN (available as PARAMS)
{_dump(plan, 4000)}

PROFILE OF THE CLEANED DATA (INPUT_PATH is the cleaned CSV)
{_dump(cleaned_profile)}

METHOD
- Treat numeric columns with > 10 distinct values as numeric, everything else as categorical.
- Classification target (1 = event): numeric features -> Mann-Whitney U (and Cohen's d);
  categorical -> chi-square + Cramer's V and the event rate per category.
  Regression target: numeric -> Spearman; categorical -> Kruskal-Wallis + eta.
- Correct all p-values for multiple testing (Benjamini-Hochberg) and report effect sizes, not just p.
- Put numeric and categorical effects on a common 0-1 "strength" scale so they can be ranked.

REQUIRED OUTPUTS
1. save_json({{...}}, "stats.json") with keys:
   task_type, target, n_rows (int), base_rate (float, classification only),
   top_drivers: list (max 10, strongest first) of dicts with keys
     "feature", "kind" ("numeric"|"categorical"), "strength" (0-1), "strength_label"
     ("negligible"|"small"|"medium"|"large"), "p_adjusted", "significant" (bool),
     "finding" (one plain-English sentence that quotes the key numbers),
   all_tests: list of every test performed (feature, test name, statistic, p_value, p_adjusted),
   charts: list of the PNG file names you saved.
2. At least 3 charts via save_fig(fig, "name.png"): target distribution, driver strength ranking
   (horizontal bars), and the top drivers (box plots / rate-by-category bars). Titles must state
   what the chart shows; label axes; use two colours at most; no 3D, no pie charts.
Do not train predictive models here.{_repair_block(previous_code, error)}"""
    return system, user


def reporter_prompts(question: str, facts: dict, feedback: str | None) -> tuple[str, str]:
    system = (
        "You are the Reporter agent: a senior analyst who writes crisp business reports for executives. "
        "You only state numbers that appear in the FACTS you are given. You never invent statistics, "
        "never claim causation from correlation, and you always mention limitations."
    )
    fix = f"\n\nREVISION NEEDED\n{feedback}\n" if feedback else ""
    user = f"""QUESTION FROM THE BUSINESS
{facts.get("question")}

FACTS (JSON, the only source of truth)
{_dump(facts, 24000)}

Write the report body in Markdown with EXACTLY these sections (use '##' headings):
## Executive summary        (3-5 bullets: the answer first, with the most important numbers)
## Data quality & preparation (what was fixed and why it matters)
## What drives the outcome   (rank the drivers; explain each in plain language with its numbers)
## Predictive model          (how well it works; what it relies on; what the metrics mean for the business)
## Recommendations           (3-5 concrete, prioritised actions tied to the evidence)
## Limitations & next steps  (correlation vs causation, data caveats, what to test next)

RULES
- Copy numbers exactly as they appear in FACTS (same rounding). Do NOT compute new numbers
  (no differences, ratios or percentages that are not already in FACTS).
- Do not embed images or tables; charts and appendices are added automatically.
- Do not add a title or any text outside the six sections. Keep it under 650 words.{fix}"""
    return system, user
