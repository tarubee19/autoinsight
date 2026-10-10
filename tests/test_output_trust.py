"""Everything a sandboxed program writes is untrusted input to the host.

The API unpickles ``model.joblib`` and serves artifacts to users, so the files that leave the sandbox must be
limited to what each step is allowed to produce - and must never be symlinks that resolve on the host.
"""

import hashlib
import json
import os

import pytest

from autoinsight import templates
from autoinsight.agents.code_agent import _promote
from autoinsight.pipeline import run_pipeline
from autoinsight.sandbox import Sandbox, sanitize_output

from .conftest import make_settings
from .helpers import ENGINEER_TEMPLATE, ScriptedLLM, fenced


def _write(root, rel, content="x"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_sanitize_output_removes_symlinks_and_never_follows_them(tmp_path):
    secret_dir = tmp_path / "host"
    secret = _write(secret_dir, "secret.txt", "top secret")
    out = tmp_path / "out"
    _write(out, "cleaned.csv", "a\n1\n")
    (out / "leak.png").symlink_to(secret)
    (out / "dirlink").symlink_to(secret_dir, target_is_directory=True)
    (out / "charts").mkdir()
    (out / "charts" / "nested.png").symlink_to(secret)

    removed, failed = sanitize_output(out)

    assert sorted(removed) == ["charts/nested.png", "dirlink", "leak.png"] and failed == []
    assert sorted(p.name for p in out.rglob("*")) == ["charts", "cleaned.csv"]
    assert secret.read_text() == "top secret"  # the target outside the sandbox is untouched


def test_promote_accepts_only_what_each_agent_owns(tmp_path):
    out = tmp_path / "out"
    for name in ("cleaned.csv", "cleaning_log.json", "stats.json", "model.joblib", "report.md", "evil.py", "notes.txt",
                 "charts/a.png", "charts/deep/b.png", "extra_table.csv"):  # fmt: skip
        _write(out, name)

    def promote(agent):
        art = tmp_path / f"art_{agent}"
        art.mkdir()
        skipped = _promote(out, art, agent)
        return sorted(p.relative_to(art).as_posix() for p in art.rglob("*") if p.is_file()), sorted(skipped)

    engineer, engineer_skipped = promote("engineer")
    assert engineer == ["charts/a.png", "cleaned.csv", "cleaning_log.json", "extra_table.csv"]
    assert {"stats.json", "model.joblib", "report.md", "evil.py", "notes.txt", "charts/deep/b.png"} <= set(engineer_skipped)

    analyst, _ = promote("analyst")
    assert analyst == ["charts/a.png", "extra_table.csv", "stats.json"]  # cannot replace cleaned.csv or add a model

    modeler, _ = promote("modeler")
    assert "model.joblib" in modeler and "report.md" not in modeler and "evil.py" not in modeler


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analyst_cannot_overwrite_the_engineers_cleaned_data(tmp_path, small_churn):
    """save_csv() is available to agent code, so an analyst program *can* try to replace cleaned.csv."""
    sneaky = fenced(templates.source(templates.ANALYST) + "\nsave_csv(pd.DataFrame({'x': [1, 2, 3]}), 'cleaned.csv')\n")
    settings = make_settings(tmp_path, allow_unsafe_local=True)
    res = run_pipeline(
        small_churn, "Find the driving factors behind customer churn", settings=settings,
        llm=ScriptedLLM(engineer=[ENGINEER_TEMPLATE], analyst=[sneaky]), sandbox=Sandbox(settings, mode="local"),
    )  # fmt: skip
    assert res.ok, res.error
    cleaned = (res.run_dir / "artifacts" / "cleaned.csv").read_text()
    assert not cleaned.startswith("x\n"), "the analyst replaced the cleaned dataset"
    engineer_out = res.run_dir / "work" / "engineer_attempt1" / "out" / "cleaned.csv"
    if engineer_out.exists():  # work dirs are kept in tests; the promoted file must equal the engineer's
        assert _sha(engineer_out) == _sha(res.run_dir / "artifacts" / "cleaned.csv")
    events = json.loads((res.run_dir / "state.json").read_text()).get("events", [])
    assert any("ignored" in e["message"] and "cleaned.csv" in e["message"] for e in events if e["agent"] == "analyst")
