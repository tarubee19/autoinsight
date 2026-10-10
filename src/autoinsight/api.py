"""REST API (FastAPI).

    POST /jobs                       upload a CSV (or pick a sample) + a question  -> job id
    GET  /jobs/{id}                  progress, agent events, summary
    GET  /jobs/{id}/report           the report (markdown, or ?format=html)
    GET  /jobs/{id}/artifacts[/...]  charts, cleaned data, model files
    GET  /jobs/{id}/schema           the input schema of the trained model
    POST /jobs/{id}/predict          score new records with the model trained for this job
    GET  /jobs/{id}/logs             every code attempt of every agent (audit trail)

Run:  uvicorn autoinsight.api:app --port 8000      (interactive docs at /docs)
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .jobs import JobManager, JobNotFound
from .predictor import PredictionError
from .profiling import PlanError, load_dataframe
from .sandbox import SandboxUnavailable

SAMPLES = {
    "telco_churn": {
        "title": "Telecom customer churn (classification)",
        "question": "Find the driving factors behind customer churn",
    },
    "hr_attrition": {
        "title": "Employee attrition (classification)",
        "question": "Why do employees leave the company?",
    },
    "house_prices": {
        "title": "House sale prices (regression)",
        "question": "What drives the sale price of a house?",
    },
}


def _samples_dir() -> Path:
    env = os.environ.get("AUTOINSIGHT_SAMPLES_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data" / "sample"


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] | None = Field(None, description="Rows to score; send only the fields you have.")
    record: dict[str, Any] | None = Field(None, description="Shortcut for a single row.")
    explain: bool = Field(True, description="Include the top factors that moved each score.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "record": {"Contract": "Month-to-month", "tenure": 3, "InternetService": "Fiber optic", "MonthlyCharges": 89.5},
                "explain": True,
            }
        }
    }


def create_app(settings: Settings | None = None, manager: JobManager | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    retention = float(os.environ.get("AUTOINSIGHT_RETENTION_HOURS", "0") or 0)
    jobs = manager or JobManager(settings, retention_hours=retention)
    api_key = os.environ.get("AUTOINSIGHT_API_KEY")

    app = FastAPI(
        title="AutoInsight API",
        version=__version__,
        description=(
            "Upload a CSV and a business question; a team of LLM agents (Planner, Data Engineer, Analyst, "
            "Modeler, Reporter) cleans the data, runs statistics inside a sandbox, trains a model and writes a "
            "report. Then score new records with the trained model."
        ),
    )

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if api_key and not (x_api_key and secrets.compare_digest(x_api_key, api_key)):
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")

    auth = [Depends(require_key)]

    def job_or_404(job_id: str) -> Path:
        try:
            return jobs.run_path(job_id)
        except JobNotFound:
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'.") from None

    # -- meta -----------------------------------------------------------------------------------------
    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {
            "status": "ok", "version": __version__, "llm": jobs.llm.name, "sandbox": jobs.sandbox.mode,
            "auth_required": bool(api_key),
        }  # fmt: skip

    @app.get("/samples", tags=["meta"], dependencies=auth)
    def samples() -> list[dict]:
        base = _samples_dir()
        return [{"name": k, **v} for k, v in SAMPLES.items() if (base / f"{k}.csv").is_file()]

    # -- jobs -------------------------------------------------------------------------------------------
    @app.post("/jobs", status_code=202, tags=["jobs"], dependencies=auth)
    async def create_job(
        question: str = Form(..., min_length=3, max_length=500, description="Business question, e.g. 'Find the driving factors behind customer churn'"),
        file: UploadFile | None = File(None, description="CSV file"),
        sample: str | None = Form(None, description="Name of a built-in sample dataset instead of a file"),
        target: str | None = Form(None, description="Optional: exact name of the column to explain/predict"),
    ) -> dict:
        if bool(file) == bool(sample):
            raise HTTPException(422, "Provide exactly one of 'file' or 'sample'.")
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / "upload.csv"
            if sample:
                src = _samples_dir() / f"{sample}.csv"
                if sample not in SAMPLES or not src.is_file():
                    raise HTTPException(404, f"Unknown sample '{sample}'.")
                staged.write_bytes(src.read_bytes())
                name = f"{sample}.csv"
            else:
                limit = settings.max_upload_mb * 1024 * 1024
                size = 0
                with staged.open("wb") as fh:
                    while chunk := await file.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB.")
                        fh.write(chunk)
                if size == 0:
                    raise HTTPException(422, "The uploaded file is empty.")
                name = Path(file.filename or "upload.csv").name[:80]
            try:
                df = load_dataframe(staged)
            except PlanError as exc:
                raise HTTPException(422, str(exc)) from exc
            if df.shape[0] < 30 or df.shape[1] < 2:
                raise HTTPException(422, f"Need at least 30 rows and 2 columns; got {df.shape[0]} x {df.shape[1]}.")
            try:
                job_id = jobs.submit(staged, question.strip(), (target or "").strip() or None, original_name=name)
            except SandboxUnavailable as exc:
                raise HTTPException(503, str(exc)) from exc
        return {"job_id": job_id, "status": "queued"}

    @app.get("/jobs", tags=["jobs"], dependencies=auth)
    def list_jobs(limit: int = Query(30, ge=1, le=100)) -> list[dict]:
        return jobs.list_jobs(limit)

    @app.get("/jobs/{job_id}", tags=["jobs"], dependencies=auth)
    def job_status(job_id: str) -> dict:
        job_or_404(job_id)
        return jobs.status(job_id)

    @app.delete("/jobs/{job_id}", status_code=204, tags=["jobs"], dependencies=auth)
    def delete_job(job_id: str) -> None:
        job_or_404(job_id)
        jobs.delete(job_id)

    @app.get("/jobs/{job_id}/report", tags=["results"], dependencies=auth)
    def report(job_id: str, format: str = Query("md", pattern="^(md|html)$")):
        art = job_or_404(job_id) / "artifacts"
        path = art / ("report.html" if format == "html" else "report.md")
        if not path.is_file():
            raise HTTPException(409, "The report is not ready yet.")
        text = path.read_text(encoding="utf-8")
        return HTMLResponse(text) if format == "html" else PlainTextResponse(text, media_type="text/markdown; charset=utf-8")

    @app.get("/jobs/{job_id}/artifacts", tags=["results"], dependencies=auth)
    def list_artifacts(job_id: str) -> list[str]:
        art = job_or_404(job_id) / "artifacts"
        return sorted(str(p.relative_to(art)) for p in art.rglob("*") if p.is_file()) if art.is_dir() else []

    @app.get("/jobs/{job_id}/artifacts/{path:path}", tags=["results"], dependencies=auth)
    def get_artifact(job_id: str, path: str):
        art = (job_or_404(job_id) / "artifacts").resolve()
        target = (art / path).resolve()
        if art not in target.parents or not target.is_file():
            raise HTTPException(404, "No such artifact.")
        return FileResponse(target)

    @app.get("/jobs/{job_id}/logs", tags=["results"], dependencies=auth)
    def agent_logs(job_id: str) -> list[dict]:
        logs = job_or_404(job_id) / "logs"
        items = []
        for path in sorted(logs.glob("*_attempt*.json")) if logs.is_dir() else []:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        return items

    # -- model -------------------------------------------------------------------------------------------
    @app.get("/jobs/{job_id}/schema", tags=["model"], dependencies=auth)
    def schema(job_id: str) -> dict:
        job_or_404(job_id)
        try:
            return jobs.predictor(job_id).public_schema()
        except FileNotFoundError:
            raise HTTPException(404, "No model was trained for this job.") from None

    @app.post("/jobs/{job_id}/predict", tags=["model"], dependencies=auth)
    def predict(job_id: str, body: PredictRequest) -> dict:
        job_or_404(job_id)
        records = body.records or ([body.record] if body.record else None)
        if not records:
            raise HTTPException(422, "Send 'record' (one row) or 'records' (a list of rows).")
        try:
            predictor = jobs.predictor(job_id)
        except FileNotFoundError:
            raise HTTPException(404, "No model was trained for this job.") from None
        try:
            results = predictor.predict(records, explain=body.explain)
        except PredictionError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"job_id": job_id, "task_type": predictor.schema["task_type"], "target": predictor.schema["target"], "model": predictor.schema["model"], "predictions": results}

    return app


app = create_app()
