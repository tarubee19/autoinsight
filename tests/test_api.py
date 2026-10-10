import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from autoinsight.api import create_app
from autoinsight.jobs import JobManager

from .conftest import make_settings


@pytest.fixture(scope="module")
def client(tmp_root):
    settings = make_settings(tmp_root / "api_module")
    return TestClient(create_app(settings, JobManager(settings)))


def wait(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get(f"/jobs/{job_id}").json()
        if state["status"] in ("completed", "failed"):
            return state
        time.sleep(0.5)
    raise AssertionError("job did not finish")


@pytest.fixture(scope="module")
def finished_job(client, small_churn):
    with small_churn.open("rb") as fh:
        resp = client.post("/jobs", data={"question": "Find the drivers of customer churn"}, files={"file": ("churn.csv", fh, "text/csv")})
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    state = wait(client, job_id)
    assert state["status"] == "completed", state
    return job_id


def test_health_and_samples(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["llm"].startswith("offline")
    assert {s["name"] for s in client.get("/samples").json()} == {"telco_churn", "hr_attrition", "house_prices"}


def test_job_lifecycle_and_report(client, finished_job):
    state = client.get(f"/jobs/{finished_job}").json()
    assert state["summary"]["target"] == "Churn" and state["filename"] == "churn.csv"
    assert set(state["stages"].values()) == {"done"}
    md = client.get(f"/jobs/{finished_job}/report")
    assert md.status_code == 200 and md.text.startswith("# Automated Data Analysis Report")
    assert "<html" in client.get(f"/jobs/{finished_job}/report", params={"format": "html"}).text
    assert "report.md" in client.get(f"/jobs/{finished_job}/artifacts").json()
    png = client.get(f"/jobs/{finished_job}/artifacts/charts/driver_strength.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    logs = client.get(f"/jobs/{finished_job}/logs").json()
    assert {log["agent"] for log in logs} >= {"engineer", "analyst", "modeler"}
    assert any(j["run_id"] == finished_job for j in client.get("/jobs").json())


def test_predict_endpoint(client, finished_job):
    schema = client.get(f"/jobs/{finished_job}/schema").json()
    assert schema["target"] == "Churn" and {f["name"] for f in schema["features"]} >= {"Contract", "tenure"}
    risky = {"Contract": "Month-to-month", "tenure": 2, "InternetService": "Fiber optic", "MonthlyCharges": 95, "TechSupport": "No"}
    safe = {"Contract": "Two year", "tenure": 65, "InternetService": "DSL", "MonthlyCharges": 45, "TechSupport": "Yes"}
    body = client.post(f"/jobs/{finished_job}/predict", json={"records": [risky, safe]}).json()
    a, b = body["predictions"]
    assert body["task_type"] == "classification"
    assert a["probability"] > 0.5 > b["probability"] and a["risk_band"] == "high" and b["risk_band"] == "low"
    assert a["top_factors"][0]["feature"] == "Contract" and a["top_factors"][0]["effect"] > 0
    assert "defaulted_fields" in a["warnings"]


def test_predict_validation(client, finished_job):
    url = f"/jobs/{finished_job}/predict"
    assert client.post(url, json={}).status_code == 422
    assert client.post(url, json={"record": {"tenure": "abc"}}).status_code == 422
    odd = client.post(url, json={"record": {"Contract": "Weekly", "tenure": 5000, "bogus": 1}}).json()["predictions"][0]["warnings"]
    assert odd["unseen_categories"] and odd["out_of_range"] and odd["ignored_fields"] == ["bogus"]


def test_upload_validation(client):
    assert client.post("/jobs", data={"question": "find drivers"}).status_code == 422                       # no data
    assert client.post("/jobs", data={"question": "find drivers", "sample": "nope"}).status_code == 404
    both = client.post("/jobs", data={"question": "find drivers", "sample": "telco_churn"}, files={"file": ("a.csv", b"a,b\n1,2\n")})
    assert both.status_code == 422
    assert client.post("/jobs", data={"question": "find drivers"}, files={"file": ("a.csv", b"")}).status_code == 422
    tiny = client.post("/jobs", data={"question": "find drivers"}, files={"file": ("a.csv", b"a,b\n1,2\n")})
    assert tiny.status_code == 422 and "at least 30 rows" in tiny.json()["detail"]
    assert client.post("/jobs", data={"question": "x"}, files={"file": ("a.csv", b"a,b\n")}).status_code == 422  # question too short


def test_oversized_upload_is_rejected(tmp_root):
    settings = make_settings(tmp_root / "api_small", max_upload_mb=1)
    client = TestClient(create_app(settings, JobManager(settings)))
    big = io.BytesIO(b"a,b\n" + b"1,2\n" * 400_000)
    assert client.post("/jobs", data={"question": "find drivers"}, files={"file": ("big.csv", big)}).status_code == 413


def test_path_traversal_and_unknown_ids(client, finished_job):
    assert client.get("/jobs/does-not-exist").status_code == 404
    assert client.get("/jobs/20260101-000000-abcdef").status_code == 404
    assert client.get(f"/jobs/{finished_job}/artifacts/../../upload.csv").status_code == 404
    assert client.get(f"/jobs/{finished_job}/artifacts/%2e%2e/%2e%2e/state.json").status_code == 404
    assert client.get("/jobs/..%2f..%2fetc/report").status_code == 404


def test_failed_job_is_reported_not_raised(client, tmp_path):
    rows = pd.DataFrame({"a": range(60), "b": ["x", "y", "z"] * 20})
    buf = io.StringIO()
    rows.to_csv(buf, index=False)
    job = client.post("/jobs", data={"question": "analyze the data"}, files={"file": ("x.csv", buf.getvalue().encode())}).json()["job_id"]
    state = wait(client, job)
    assert state["status"] == "failed" and "Could not tell which column" in state["error"]
    assert client.post(f"/jobs/{job}/predict", json={"record": {"a": 1}}).status_code == 404
    assert client.get(f"/jobs/{job}/schema").status_code == 404


def test_api_key_protection(tmp_root, monkeypatch):
    monkeypatch.setenv("AUTOINSIGHT_API_KEY", "s3cret")
    settings = make_settings(tmp_root / "api_auth")
    client = TestClient(create_app(settings, JobManager(settings)))
    assert client.get("/health").status_code == 200  # health stays open for load balancers
    assert client.get("/jobs").status_code == 401
    assert client.get("/jobs", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/jobs", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_delete_job(client, tmp_root, small_churn):
    with small_churn.open("rb") as fh:
        job = client.post("/jobs", data={"question": "drivers of churn"}, files={"file": ("c.csv", fh)}).json()["job_id"]
    wait(client, job)
    assert client.delete(f"/jobs/{job}").status_code == 204
    assert client.get(f"/jobs/{job}").status_code == 404
