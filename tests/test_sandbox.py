import shutil

import pandas as pd
import pytest

from autoinsight.config import Settings
from autoinsight.sandbox import Sandbox, SandboxUnavailable, build_docker_command, docker_ready


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "in.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(path, index=False)
    return path


@pytest.fixture
def sandbox():
    return Sandbox(Settings(sandbox_mode="local"), mode="local")


def test_runs_code_and_collects_outputs(sandbox, csv, tmp_path):
    code = "import pandas as pd\ndf = pd.read_csv(INPUT_PATH)\nsave_json({'rows': len(df), 'p': PARAMS['k']}, 'o.json')\nlog('hi')\n"
    res = sandbox.run(code, input_csv=csv, params={"k": 7}, exec_dir=tmp_path / "e")
    assert res.ok and res.stdout.strip() == "hi"
    assert "o.json" in res.files
    assert '"rows": 3' in (res.out_dir / "o.json").read_text()


def test_traceback_points_at_agent_code(sandbox, csv, tmp_path):
    res = sandbox.run("x = 1\ny = x / 0\n", input_csv=csv, params={}, exec_dir=tmp_path / "e")
    assert not res.ok
    assert "line 2" in res.error_text() and "ZeroDivisionError" in res.error_text()


def test_timeout_is_enforced(sandbox, csv, tmp_path):
    res = sandbox.run("while True:\n    pass\n", input_csv=csv, params={}, exec_dir=tmp_path / "e", timeout_s=2)
    assert not res.ok and res.timed_out and "timed out" in res.error_text()


def test_memory_limit_is_enforced(csv, tmp_path):
    sb = Sandbox(Settings(sandbox_mode="local", sandbox_memory_mb=512), mode="local")
    res = sb.run("x = bytearray(6 * 1024**3)\n", input_csv=csv, params={}, exec_dir=tmp_path / "e")
    assert not res.ok and "MemoryError" in res.error_text()


def test_non_finite_numbers_become_valid_json(sandbox, csv, tmp_path):
    res = sandbox.run("import numpy as np\nsave_json({'a': np.nan, 'b': np.float64(1.5), 'c': np.int64(2)}, 'o')\n", input_csv=csv, params={}, exec_dir=tmp_path / "e")
    assert res.ok
    assert (res.out_dir / "o.json").read_text().count("null") == 1


def test_output_filenames_cannot_escape(sandbox, csv, tmp_path):
    res = sandbox.run("save_json({'x': 1}, '../../escape.json')\n", input_csv=csv, params={}, exec_dir=tmp_path / "e")
    assert res.ok and "escape.json" in res.files
    assert not (tmp_path / "escape.json").exists()


def test_llm_code_is_refused_in_weak_sandbox():
    sb = Sandbox(Settings(sandbox_mode="local"), mode="local")
    with pytest.raises(SandboxUnavailable):
        sb.assert_safe_for(llm_is_offline=False)
    sb.assert_safe_for(llm_is_offline=True)
    Sandbox(Settings(sandbox_mode="local", allow_unsafe_local=True), mode="local").assert_safe_for(llm_is_offline=False)


def test_docker_mode_without_docker_is_an_explicit_error():
    if docker_ready("autoinsight-sandbox:latest"):
        pytest.skip("docker is available here")
    with pytest.raises(SandboxUnavailable):
        Sandbox(Settings(sandbox_mode="docker"), mode="docker")


def test_docker_command_is_hardened():
    cmd = " ".join(build_docker_command(image="img", in_dir="/i", out_dir="/o", name="n", memory_mb=1024))
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges", "--user 65534:65534",
                 "--pids-limit", "--memory 1024m", "-v /i:/in:ro", "-v /o:/out:rw"):
        assert flag in cmd
    assert "--privileged" not in cmd and "docker.sock" not in cmd


@pytest.mark.docker
@pytest.mark.skipif(not shutil.which("docker") or not docker_ready("autoinsight-sandbox:latest"), reason="needs Docker + sandbox image")
def test_docker_backend_runs_code_without_network(csv, tmp_path):
    sb = Sandbox(Settings(sandbox_mode="docker"), mode="docker")
    code = "import pandas as pd\ndf = pd.read_csv(INPUT_PATH)\nsave_json({'rows': len(df)}, 'o.json')\n"
    res = sb.run(code, input_csv=csv, params={}, exec_dir=tmp_path / "e")
    assert res.ok, res.stderr
    # network must be unreachable from inside the container
    probe = "import socket\ns = socket.socket()\ns.settimeout(3)\ns.connect(('1.1.1.1', 80))\n"
    res = sb.run(probe, input_csv=csv, params={}, exec_dir=tmp_path / "e2")
    assert not res.ok
