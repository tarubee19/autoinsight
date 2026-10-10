"""Background job management for the API (thread pool + filesystem as the source of truth).

State lives in ``runs/<id>/state.json`` / ``summary.json`` - not in memory - so the API can be
restarted without losing finished runs, and any worker/process can answer a status request.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .llm import LLMClient, build_llm
from .pipeline import new_run_id, run_pipeline, write_state
from .predictor import Predictor
from .sandbox import Sandbox
from .state import STAGES

RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")


class JobNotFound(KeyError):
    pass


class JobManager:
    def __init__(self, settings: Settings | None = None, max_workers: int = 2, retention_hours: float = 0):
        self.settings = settings or Settings.from_env()
        self.settings.runs_dir.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="autoinsight")
        self.retention_hours = retention_hours
        self._predictors: dict[str, Predictor] = {}
        self._lock = threading.Lock()
        self._llm: LLMClient | None = None
        self._sandbox: Sandbox | None = None

    # -- shared, lazily built components ---------------------------------------------------------
    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    @property
    def sandbox(self) -> Sandbox:
        if self._sandbox is None:
            self._sandbox = Sandbox(self.settings)
        return self._sandbox

    # -- paths ---------------------------------------------------------------------------------------
    def run_path(self, job_id: str) -> Path:
        """Validated path of a run directory (the id regex prevents path traversal)."""
        if not RUN_ID.match(job_id or ""):
            raise JobNotFound(job_id)
        path = (self.settings.runs_dir / job_id).resolve()
        if not path.is_dir() or self.settings.runs_dir.resolve() not in path.parents:
            raise JobNotFound(job_id)
        return path

    # -- lifecycle -----------------------------------------------------------------------------------
    def submit(self, csv_path: Path, question: str, target: str | None = None, original_name: str = "upload.csv") -> str:
        self.purge_expired()
        job_id = new_run_id()
        run_dir = self.settings.runs_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        staged = run_dir / "upload.csv"
        shutil.copyfile(csv_path, staged)
        write_state(run_dir, {
            "run_id": job_id, "status": "queued", "error": None, "question": question, "stages": {s: "pending" for s in STAGES},
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"), "elapsed_s": 0, "llm": self.llm.name,
            "sandbox": self.sandbox.mode, "events": [], "filename": original_name,
        })  # fmt: skip
        self.pool.submit(self._run, job_id, staged, question, target, original_name)
        return job_id

    def _run(self, job_id: str, staged: Path, question: str, target: str | None, original_name: str) -> None:
        try:
            run_pipeline(staged, question, target=target, settings=self.settings, llm=self.llm, sandbox=self.sandbox, run_id=job_id, state_extra={"filename": original_name})
        except Exception as exc:  # noqa: BLE001 - e.g. SandboxUnavailable (safety gate)
            run_dir = self.settings.runs_dir / job_id
            state_path = run_dir / "state.json"
            state = json.loads(state_path.read_text()) if state_path.is_file() else {"run_id": job_id}
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_state(run_dir, state)
            (run_dir / "summary.json").write_text(json.dumps({"status": "failed", "error": state["error"]}), encoding="utf-8")
        finally:
            (staged).unlink(missing_ok=True)

    def status(self, job_id: str) -> dict:
        run_dir = self.run_path(job_id)
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            state["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        return state

    def list_jobs(self, limit: int = 30) -> list[dict]:
        rows = []
        for path in sorted(self.settings.runs_dir.iterdir(), reverse=True):
            if path.is_dir() and RUN_ID.match(path.name) and (path / "state.json").is_file():
                try:
                    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                rows.append({k: state.get(k) for k in ("run_id", "status", "question", "started_at", "elapsed_s", "filename")})
            if len(rows) >= limit:
                break
        return rows

    def delete(self, job_id: str) -> None:
        run_dir = self.run_path(job_id)
        with self._lock:
            self._predictors.pop(job_id, None)
        shutil.rmtree(run_dir, ignore_errors=True)

    def purge_expired(self) -> int:
        """Delete runs older than ``retention_hours`` (privacy for public demos). 0 = keep forever."""
        if not self.retention_hours:
            return 0
        cutoff, removed = time.time() - self.retention_hours * 3600, 0
        for path in self.settings.runs_dir.iterdir():
            if path.is_dir() and RUN_ID.match(path.name) and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed

    # -- predictions -----------------------------------------------------------------------------------
    def predictor(self, job_id: str) -> Predictor:
        run_dir = self.run_path(job_id)
        with self._lock:
            if job_id not in self._predictors:
                self._predictors[job_id] = Predictor.load(run_dir / "artifacts")
            return self._predictors[job_id]
