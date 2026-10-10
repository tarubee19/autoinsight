"""Shared fixtures. Environment is pinned *before* autoinsight is imported so that nothing from the
developer's shell (API keys, runs dir) leaks into the tests."""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="autoinsight-tests-"))
for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(key, None)
os.environ["AUTOINSIGHT_LLM_PROVIDER"] = "offline"
os.environ["AUTOINSIGHT_SANDBOX_MODE"] = "local"
os.environ["AUTOINSIGHT_RUNS_DIR"] = str(_TMP / "api_runs")
os.environ["AUTOINSIGHT_SAMPLES_DIR"] = str(Path(__file__).resolve().parents[1] / "data" / "sample")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from autoinsight.config import Settings  # noqa: E402
from autoinsight.pipeline import run_pipeline  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "sample"


def make_settings(tmp: Path, **overrides) -> Settings:
    base = dict(llm_provider="offline", sandbox_mode="local", runs_dir=tmp / "runs", sandbox_timeout_s=120)
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session")
def samples() -> Path:
    return SAMPLES


@pytest.fixture(scope="session")
def small_churn(tmp_root) -> Path:
    """800-row slice of the churn sample (fast) with its mess intact."""
    df = pd.read_csv(SAMPLES / "telco_churn.csv").head(900)
    path = tmp_root / "churn_small.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def churn_run(tmp_root):
    """One full offline run on the complete churn sample, shared by many tests."""
    settings = make_settings(tmp_root / "churn_full")
    return run_pipeline(SAMPLES / "telco_churn.csv", "Find the driving factors behind customer churn", settings=settings)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)
