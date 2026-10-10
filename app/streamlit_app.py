"""Streamlit front end for the AutoInsight API.

    streamlit run app/streamlit_app.py            (API must be running: autoinsight serve)

Environment: API_URL (default http://localhost:8000), AUTOINSIGHT_API_KEY (if the API needs one).
"""

from __future__ import annotations

import io
import os
import re
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
HEADERS = {"X-API-Key": os.environ["AUTOINSIGHT_API_KEY"]} if os.environ.get("AUTOINSIGHT_API_KEY") else {}
STAGE_LABELS = {
    "profile": "Profile", "planner": "Planner", "engineer": "Data Engineer",
    "analyst": "Analyst", "modeler": "Modeler", "reporter": "Reporter",
}  # fmt: skip
STAGE_ICON = {"pending": "⚪", "running": "🟡", "done": "🟢", "failed": "🔴", "skipped": "⚫"}

st.set_page_config(page_title="AutoInsight", page_icon="🔎", layout="wide")


# ------------------------------------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------------------------------------
class ApiError(RuntimeError):
    pass


def call(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", headers=HEADERS, timeout=kwargs.pop("timeout", 60), **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the API at {API_URL}: {exc}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(f"{resp.status_code}: {detail}")
    return resp


def get_json(path: str):
    return call("GET", path).json()


# ------------------------------------------------------------------------------------------------
# Rendering helpers
# ------------------------------------------------------------------------------------------------
def render_report(job_id: str, markdown_text: str) -> None:
    """Render the markdown report; charts are fetched from the API and shown as native images."""
    parts = re.split(r"!\[[^\]]*\]\((charts/[^)]+)\)", markdown_text)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            try:
                st.image(call("GET", f"/jobs/{job_id}/artifacts/{part}").content, width="stretch")
            except ApiError:
                st.caption(f"(chart {part} unavailable)")


def stage_tracker(state: dict) -> None:
    cols = st.columns(len(STAGE_LABELS))
    for col, (key, label) in zip(cols, STAGE_LABELS.items(), strict=True):
        col.markdown(f"{STAGE_ICON.get(state['stages'].get(key, 'pending'), '⚪')} **{label}**")


def event_log(state: dict) -> None:
    icon = {"ok": "✅", "retry": "🔁", "failed": "❌", "skipped": "⏭️"}
    for e in state.get("events", []):
        st.write(f"{icon.get(e['status'], '•')} **{e['agent']}** - {e['message']}")


def prediction_form(job_id: str, schema: dict) -> None:
    st.caption(
        f"Fill in a record to score it with the **{schema['model']}** trained on this dataset "
        f"(target: `{schema['target']}`). Fields you leave at their defaults use the typical training value."
    )
    values: dict = {}
    cols = st.columns(3)
    for i, feat in enumerate(schema["features"]):
        box = cols[i % 3]
        key = f"pred_{job_id}_{feat['name']}"
        if feat["kind"] == "numeric" and not feat.get("choices"):
            span = (feat["max"] - feat["min"]) or 1.0
            lo, hi = feat["min"] - 0.25 * span, feat["max"] + 0.25 * span
            step = 1.0 if feat.get("integer") else max(span / 100, 0.01)
            values[feat["name"]] = box.number_input(
                feat["name"], min_value=float(lo), max_value=float(hi), value=float(feat["default"]), step=float(step), key=key,
                help=f"training range {feat['min']:g} - {feat['max']:g}",
            )  # fmt: skip
        else:
            choices = feat["choices"]
            default = feat["default"]
            index = choices.index(default) if default in choices else 0
            values[feat["name"]] = box.selectbox(feat["name"], choices, index=index, key=key)
    if st.button("Predict", type="primary", key=f"predict_{job_id}"):
        try:
            result = call("POST", f"/jobs/{job_id}/predict", json={"record": values, "explain": True}).json()["predictions"][0]
        except ApiError as exc:
            st.error(str(exc))
            return
        if schema["task_type"] == "classification":
            m1, m2, m3 = st.columns(3)
            m1.metric(f"P({schema['target']} = {schema['positive_label']})", f"{result['probability']:.1%}")
            m2.metric("Risk band", result["risk_band"].upper(), help=f"high >= {result['threshold']:.0%}, medium >= base rate {result['base_rate']:.0%}")
            m3.metric("Decision", result["label"])
            st.progress(min(max(result["probability"], 0.0), 1.0))
        else:
            st.metric(f"Predicted {schema['target']}", f"{result['prediction']:,.2f}")
        if result.get("top_factors"):
            st.markdown("**What moved this score** (vs a typical value of each feature; positive = pushes the score up)")
            factors = pd.DataFrame(result["top_factors"])
            factors["label"] = factors["feature"] + " = " + factors["value"].astype(str)
            st.bar_chart(factors.set_index("label")["effect"], horizontal=True)
        if result.get("warnings"):
            with st.expander("Input warnings"):
                st.json(result["warnings"])


# ------------------------------------------------------------------------------------------------
# Page
# ------------------------------------------------------------------------------------------------
st.title("🔎 AutoInsight")
st.caption("Upload a CSV, ask a business question - a team of AI agents cleans the data, runs the statistics, trains a model and writes the report.")

try:
    health = get_json("/health")
except ApiError as exc:
    st.error(str(exc))
    st.info("Start the API first:  `autoinsight serve`  (or `docker compose up`).")
    st.stop()

with st.sidebar:
    st.subheader("Engine")
    st.write(f"**LLM:** `{health['llm']}`")
    st.write(f"**Sandbox:** `{health['sandbox']}`")
    if health["llm"].startswith("offline"):
        st.info("Offline mode: vetted code templates, no API key needed. Set an API key to let LLM agents write the code.")
    st.divider()
    st.subheader("Previous runs")
    runs = get_json("/jobs?limit=15")
    labels = {r["run_id"]: f"{r['run_id'][:15]} · {r.get('status')} · {(r.get('filename') or '')[:18]}" for r in runs}
    pick = st.selectbox("Open a run", [""] + list(labels), format_func=lambda k: labels.get(k, "-"))
    if pick and st.button("Open"):
        st.session_state["job_id"] = pick
        st.rerun()

with st.expander("New analysis", expanded="job_id" not in st.session_state):
    samples = get_json("/samples")
    source = st.radio("Data", ["Sample dataset", "Upload CSV"], horizontal=True)
    sample_name, upload = None, None
    if source == "Sample dataset" and samples:
        sample_name = st.selectbox("Sample", [s["name"] for s in samples], format_func=lambda n: next(s["title"] for s in samples if s["name"] == n))
        default_q = next(s["question"] for s in samples if s["name"] == sample_name)
    else:
        upload = st.file_uploader("CSV file", type=["csv"])
        default_q = ""
    question = st.text_input("Business question", value=default_q, key=f"q_{sample_name}", placeholder="Find the driving factors behind customer churn")
    target = st.text_input("Target column (optional)", help="Leave empty to let the Planner agent decide from your question.")
    if st.button("Run analysis", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif source == "Upload CSV" and upload is None:
            st.warning("Please upload a CSV file.")
        else:
            try:
                data = {"question": question, "target": target}
                files = None
                if sample_name:
                    data["sample"] = sample_name
                else:
                    files = {"file": (upload.name, io.BytesIO(upload.getvalue()), "text/csv")}
                st.session_state["job_id"] = call("POST", "/jobs", data=data, files=files).json()["job_id"]
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))

job_id = st.session_state.get("job_id")
if job_id:
    try:
        state = get_json(f"/jobs/{job_id}")
    except ApiError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader(f"Run `{job_id}`")
    st.caption(f"{state.get('question')}  ·  {state.get('elapsed_s', 0):.0f}s")
    stage_tracker(state)

    if state["status"] in ("queued", "running"):
        done = sum(1 for s in state["stages"].values() if s in ("done", "skipped"))
        st.progress(done / len(state["stages"]), text="Agents are working...")
        with st.expander("Live agent log", expanded=True):
            event_log(state)
        time.sleep(1.2)
        st.rerun()

    elif state["status"] == "failed":
        st.error(state.get("error") or (state.get("summary") or {}).get("error") or "The run failed.")
        with st.expander("Agent log", expanded=True):
            event_log(state)

    else:
        summary = state["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target", summary["target"])
        c2.metric("Task", summary["task_type"])
        model = summary.get("model") or {}
        if "roc_auc" in model:
            c3.metric("Model ROC AUC", f"{model['roc_auc']:.3f}")
            c4.metric("Recall", f"{model['recall']:.2f}")
        elif "r2" in model:
            c3.metric("Model R²", f"{model['r2']:.3f}")
            c4.metric("MAE", f"{model['mae']:,.0f}")
        g = summary.get("grounding") or {}
        st.caption(f"Top drivers: {', '.join(summary['top_drivers'])} · numbers verified against computed results: {g.get('numbers_grounded', 0)}/{g.get('numbers_checked', 0)}")

        tab_report, tab_predict, tab_data, tab_logs = st.tabs(["📄 Report", "🎯 Predict", "🗂 Data & files", "🧾 Agent logs"])
        with tab_report:
            md = call("GET", f"/jobs/{job_id}/report").text
            d1, d2, _ = st.columns([1, 1, 4])
            d1.download_button("Download .md", md, file_name=f"report_{job_id}.md")
            d2.download_button("Download .html", call("GET", f"/jobs/{job_id}/report", params={"format": "html"}).text, file_name=f"report_{job_id}.html")
            render_report(job_id, md)
        with tab_predict:
            try:
                prediction_form(job_id, get_json(f"/jobs/{job_id}/schema"))
            except ApiError:
                st.info("No model was trained for this run (see the report's limitations).")
        with tab_data:
            try:
                cleaned = pd.read_csv(io.BytesIO(call("GET", f"/jobs/{job_id}/artifacts/cleaned.csv").content), keep_default_na=False, na_values=[""])
                st.markdown(f"**Cleaned dataset** - {len(cleaned):,} rows × {cleaned.shape[1]} columns")
                st.dataframe(cleaned.head(200), width="stretch")
            except ApiError:
                pass
            for name in ("cleaned.csv", "cleaning_log.json", "stats.json", "metrics.json", "model.joblib"):
                try:
                    st.download_button(f"Download {name}", call("GET", f"/jobs/{job_id}/artifacts/{name}").content, file_name=name, key=f"dl_{name}")
                except ApiError:
                    pass
        with tab_logs:
            st.caption("Every program the agents wrote and ran, including the failed attempts that were repaired.")
            for log in get_json(f"/jobs/{job_id}/logs"):
                verdict = "✅ ok" if log["ok"] else "❌ failed"
                with st.expander(f"{log['agent']} · attempt {log['attempt']} · {log['source']} · {verdict}"):
                    if log.get("error"):
                        st.error(log["error"])
                    if log.get("stdout"):
                        st.text(log["stdout"])
                    if log.get("code"):
                        st.code(log["code"], language="python")
