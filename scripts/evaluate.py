"""Benchmark the system and write docs/EVALUATION.md + docs/evaluation_results.json.

Two experiments (both run offline - no API key, fully reproducible):

1. DRIVER RECOVERY - three synthetic datasets with *planted* drivers. Did the statistical agent and
   the model rank the true drivers on top?   precision@k with k = number of planted drivers.
2. FAULT INJECTION - a scripted "LLM" misbehaves in six ways (unsafe code, runtime bug, missing
   outputs, hallucinated features, outage, invented statistics). Does the workflow still finish
   with a correct report, and how many attempts did recovery take?

Usage:
    python scripts/evaluate.py              offline mode, ~2 minutes  -> docs/EVALUATION.md
    python scripts/evaluate.py --real-llm   uses the LLM + sandbox from your environment (API key,
                                            Docker) for experiment 1 only -> docs/EVALUATION_LLM.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoinsight import __version__  # noqa: E402
from autoinsight.config import Settings  # noqa: E402
from autoinsight.pipeline import run_pipeline  # noqa: E402
from autoinsight.sandbox import Sandbox  # noqa: E402
from autoinsight.testing import (  # noqa: E402
    HALLUCINATED_FEATURE,
    INVENTED_STATS_REPORT,
    NO_OUTPUT,
    RUNTIME_BUG,
    UNSAFE_CODE,
    ScriptedLLM,
    analyst_template,
    engineer_template,
)  # fmt: skip

spec = importlib.util.spec_from_file_location("gen", ROOT / "data" / "generate_sample_data.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

QUESTIONS = {
    "telco_churn": "Find the driving factors behind customer churn",
    "hr_attrition": "Why do employees leave the company?",
    "house_prices": "What drives the sale price of a house?",
}


def settings_for(tmp: Path, **kw) -> Settings:
    return Settings(llm_provider="offline", sandbox_mode="local", runs_dir=tmp, **kw)


def driver_recovery(tmp: Path, real_llm: bool = False) -> list[dict]:
    rows = []
    for name, question in QUESTIONS.items():
        csv = ROOT / "data" / "sample" / f"{name}.csv"
        started = time.monotonic()
        settings = replace(Settings.from_env(), runs_dir=tmp / name) if real_llm else settings_for(tmp / name)
        res = run_pipeline(csv, question, settings=settings)
        elapsed = time.monotonic() - started
        art = res.run_dir / "artifacts"
        planted = set(gen.PLANTED_DRIVERS[name])
        k = len(planted)
        stats = json.loads((art / "stats.json").read_text())
        importance = json.loads((art / "feature_importance.json").read_text())["features"]
        metrics = json.loads((art / "metrics.json").read_text())
        cleaning = json.loads((art / "cleaning_log.json").read_text())
        stat_top = [d["feature"] for d in stats["top_drivers"][:k]]
        model_top = [f["feature"] for f in importance[:k]]
        df = pd.read_csv(csv)
        rows.append({
            "dataset": name, "rows": len(df), "columns": df.shape[1], "task": res.summary["task_type"],
            "metric": "ROC AUC" if "roc_auc" in metrics else "R2",
            "score": round(metrics.get("roc_auc", metrics.get("r2")), 3),
            "k": k,
            "stats_precision_at_k": round(len(planted & set(stat_top)) / k, 2),
            "model_precision_at_k": round(len(planted & set(model_top)) / k, 2),
            "stats_top": stat_top, "model_top": model_top, "planted": sorted(planted),
            "duplicates_removed": cleaning["duplicates_removed"], "columns_dropped": len(cleaning["dropped_columns"]),
            "grounding_rate": res.summary["grounding"]["grounding_rate"],
            "numbers_checked": res.summary["grounding"]["numbers_checked"],
            "seconds": round(elapsed, 1), "ok": res.ok, "llm": res.summary.get("llm"),
            "attempts": res.summary.get("attempts", {}), "fallbacks": res.summary.get("fallbacks", []),
        })  # fmt: skip
        print(f"  {name:13s} {rows[-1]['metric']} {rows[-1]['score']}  stats P@{k}={rows[-1]['stats_precision_at_k']}  model P@{k}={rows[-1]['model_precision_at_k']}  {elapsed:.0f}s")
    return rows


def fault_injection(tmp: Path) -> list[dict]:
    csv = tmp / "churn_800.csv"
    pd.read_csv(ROOT / "data" / "sample" / "telco_churn.csv").head(800).to_csv(csv, index=False)
    boom = ConnectionError("API unreachable")
    scenarios = [
        ("Unsafe code (os.system) from the Data Engineer", dict(engineer=[UNSAFE_CODE, engineer_template])),
        ("Runtime error (KeyError on a made-up column)", dict(engineer=[RUNTIME_BUG, engineer_template])),
        ("Program runs but writes no output file", dict(engineer=[NO_OUTPUT, engineer_template])),
        ("Analyst names a feature that does not exist", dict(analyst=[HALLUCINATED_FEATURE, analyst_template])),
        ("Reporter invents statistics (87.65%, 42.1%)", dict(reporter=[INVENTED_STATS_REPORT])),
        ("Every LLM call fails (API outage)", dict(planner=[boom, boom], engineer=[boom] * 3, analyst=[boom] * 3, reporter=[boom, boom])),
        ("LLM never writes working code (5 bad answers each)", dict(engineer=[RUNTIME_BUG] * 5, analyst=[UNSAFE_CODE] * 5)),
    ]
    out = []
    for label, script in scenarios:
        settings = settings_for(tmp / "fault", allow_unsafe_local=True)
        llm = ScriptedLLM(**script)
        res = run_pipeline(csv, "Find the driving factors behind customer churn", settings=settings, llm=llm, sandbox=Sandbox(settings, mode="local"))
        report = res.report_path.read_text() if res.report_path.is_file() else ""
        clean = "87.65" not in report and "42.1%" not in report and "Imaginary" not in report
        attempts = res.summary.get("attempts", {})
        out.append({
            "scenario": label, "recovered": bool(res.ok and clean), "attempts": {k: v for k, v in attempts.items() if v > 1} or "first try",
            "fallbacks": res.summary.get("fallbacks", []), "grounding_rate": (res.summary.get("grounding") or {}).get("grounding_rate"),
        })  # fmt: skip
        print(f"  {'OK ' if out[-1]['recovered'] else 'FAIL'} {label}")
    return out


def to_markdown(recovery: list[dict], faults: list[dict]) -> str:
    lines = [
        "# Evaluation",
        "",
        f"_Generated by `python scripts/evaluate.py` on {datetime.now(UTC):%Y-%m-%d} - AutoInsight {__version__}, "
        f"Python {platform.python_version()}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}. "
        "Offline mode (vetted templates), no API key, seed 42._",
        "",
        "## 1. Driver recovery on data with planted ground truth",
        "",
        "Each synthetic dataset is generated from a known process (see `data/generate_sample_data.py`) and then "
        "deliberately corrupted (duplicates, numbers stored as text, blanks, constant and ID columns, outliers, missing values). "
        "*Precision@k* = share of the k planted drivers found in the top k ranked features (k = number of planted drivers).",
        "",
        "| Dataset | Rows x cols | Task | Held-out score | Stats P@k | Model P@k | Report numbers verified | Attempts eng/ana | Fallback | Run time |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recovery:
        att = r["attempts"]
        lines.append(
            f"| `{r['dataset']}` | {r['rows']:,} x {r['columns']} | {r['task']} | {r['metric']} {r['score']} | "
            f"{r['stats_precision_at_k']:.0%} (k={r['k']}) | {r['model_precision_at_k']:.0%} | {r['grounding_rate']:.0%} of {r['numbers_checked']} | "
            f"{att.get('engineer', 1)}/{att.get('analyst', 1)} | {', '.join(r['fallbacks']) or '-'} | {r['seconds']} s |"
        )
    lines += ["", "<details><summary>Rankings per dataset</summary>", ""]
    for r in recovery:
        lines += [f"**{r['dataset']}** - planted: {', '.join(r['planted'])}", f"- statistical agent top {r['k']}: {', '.join(r['stats_top'])}", f"- model importance top {r['k']}: {', '.join(r['model_top'])}", ""]
    lines += [
        "</details>",
        "",
        "> **How to read this honestly.** These datasets are synthetic and the planted effects are fairly strong, so "
        "100% recovery shows the *pipeline works* (cleaning does not destroy signal, tests are sensible, ranking is "
        "sane) - it is not a claim about real-world insight quality. On the churn data a logistic regression wins model "
        "selection because the data was generated by a logistic process; on real data gradient boosting often wins. "
        "Run it on the public IBM Telco churn dataset or your own data for real results.",
        "",
        "## 2. Fault injection: does the workflow recover from a misbehaving LLM?",
        "",
        "A scripted LLM is made to fail in specific ways (`autoinsight.testing.ScriptedLLM`). *Recovered* = the run "
        "completed **and** the final report contains no trace of the injected falsehood.",
        "",
        "| Injected fault | Recovered | Extra attempts needed | Vetted fallback used |",
        "|---|---|---|---|",
    ]
    for f in faults:
        extra = f["attempts"] if isinstance(f["attempts"], str) else ", ".join(f"{k}: {v} attempts" for k, v in f["attempts"].items())
        lines.append(f"| {f['scenario']} | {'yes' if f['recovered'] else '**NO**'} | {extra} | {', '.join(f['fallbacks']) or '-'} |")
    lines += ["", f"**{sum(f['recovered'] for f in faults)} of {len(faults)} fault scenarios recovered.**", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-llm", action="store_true", help="run experiment 1 with the LLM/sandbox configured in the environment")
    args = parser.parse_args()
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="autoinsight-eval-") as tmp_name:
        tmp = Path(tmp_name)
        print("1/2 driver recovery" + (" (real LLM)" if args.real_llm else ""))
        recovery = driver_recovery(tmp, real_llm=args.real_llm)
        if args.real_llm:
            (docs / "EVALUATION_LLM.md").write_text(to_markdown(recovery, [], real_llm=True).split("## 2.")[0], encoding="utf-8")
            (docs / "evaluation_llm_results.json").write_text(json.dumps(recovery, indent=2), encoding="utf-8")
            print(f"wrote {docs / 'EVALUATION_LLM.md'}")
            return 0 if all(r["ok"] for r in recovery) else 1
        print("2/2 fault injection")
        faults = fault_injection(tmp)
    (docs / "evaluation_results.json").write_text(json.dumps({"driver_recovery": recovery, "fault_injection": faults}, indent=2), encoding="utf-8")
    (docs / "EVALUATION.md").write_text(to_markdown(recovery, faults), encoding="utf-8")
    print(f"wrote {docs / 'EVALUATION.md'}")
    ok = all(r["ok"] for r in recovery) and all(f["recovered"] for f in faults)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
