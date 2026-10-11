"""Isolated execution of agent-written code.

Two backends share one interface:

``docker``  (recommended; the real security boundary)
    ``docker run --network none --read-only --cap-drop ALL --user 65534 --memory .. --cpus ..
    --pids-limit ..`` with the input mounted read-only and a single writable output directory.

``local``   (development fallback, used when no Docker daemon/image is available)
    A separate Python process in isolated mode (``-I``) with ``resource`` limits (memory, CPU,
    file size) and a wiped environment. It is *not* a hard boundary, so by default it is only
    allowed to run vetted template code (offline mode). Running LLM-written code here requires the
    explicit opt-in ``AUTOINSIGHT_ALLOW_UNSAFE_LOCAL=1``.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from .config import Settings

MAX_CAPTURE = 20_000  # characters of stdout/stderr kept
MAX_OUT_BYTES = 300 * 1024 * 1024


class SandboxUnavailable(RuntimeError):
    """Raised when the requested sandbox mode cannot be provided safely."""


@dataclass
class SandboxResult:
    ok: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    mode: str
    out_dir: Path
    files: list[str] = field(default_factory=list)

    def error_text(self) -> str:
        """A compact, LLM-friendly description of what went wrong."""
        if self.timed_out:
            return f"Execution timed out after {self.duration_s:.0f}s. Make the code faster (avoid O(n^2) loops, sample large data)."
        parts = []
        if self.stderr.strip():
            parts.append(self.stderr.strip()[-3000:])
        if self.returncode not in (0, None) and not parts:
            parts.append(f"Process exited with code {self.returncode}")
        return "\n".join(parts) or "Unknown failure"


def _trim(text: str, limit: int = MAX_CAPTURE) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 4] + "\n...[truncated]...\n" + text[-(3 * limit) // 4 :]


def docker_ready(image: str) -> bool:
    """True if the Docker CLI works, the daemon answers and the sandbox image exists."""
    if not shutil.which("docker"):
        return False
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=8, check=False,
        )  # fmt: skip
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def build_docker_command(
    *, image: str, in_dir: str, out_dir: str, name: str, memory_mb: int,
    cpus: str = "1.5", pids_limit: int = 128,
) -> list[str]:  # fmt: skip
    """The exact ``docker run`` invocation (public so tests can assert the hardening flags)."""
    return [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=128m",
        "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
        "--cpus", cpus,
        "--pids-limit", str(pids_limit),
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "-v", f"{in_dir}:/in:ro",
        "-v", f"{out_dir}:/out:rw",
        "-e", "AI_IN_DIR=/in", "-e", "AI_OUT_DIR=/out",
        "-e", "MPLCONFIGDIR=/tmp/mpl", "-e", "OMP_NUM_THREADS=2",
        "-e", "OPENBLAS_NUM_THREADS=2", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-w", "/out",
        image, "python", "-I", "/in/runner.py", "agent_code.py",
    ]  # fmt: skip


class Sandbox:
    def __init__(self, settings: Settings | None = None, mode: str | None = None):
        self.settings = settings or Settings.from_env()
        requested = (mode or self.settings.sandbox_mode).lower()
        if requested == "auto":
            requested = "docker" if docker_ready(self.settings.sandbox_image) else "local"
        elif requested == "docker" and not docker_ready(self.settings.sandbox_image):
            raise SandboxUnavailable(
                f"Docker sandbox requested but Docker or image '{self.settings.sandbox_image}' is "
                "not available. Build it with: docker build -f docker/Dockerfile.sandbox "
                "-t autoinsight-sandbox:latest ."
            )
        self.mode = requested

    # -- safety gate ---------------------------------------------------------------------------
    def assert_safe_for(self, llm_is_offline: bool) -> None:
        """Refuse to run LLM-written code in the weak local sandbox unless explicitly allowed."""
        if self.mode == "local" and not llm_is_offline and not self.settings.allow_unsafe_local:
            raise SandboxUnavailable(
                "An LLM is configured, but no Docker sandbox is available. Running LLM-written "
                "code outside Docker is unsafe. Either build/start the Docker sandbox, run with "
                "AUTOINSIGHT_LLM_PROVIDER=offline, or set AUTOINSIGHT_ALLOW_UNSAFE_LOCAL=1 if you "
                "understand the risk (dev machines only)."
            )

    # -- execution -------------------------------------------------------------------------------
    def run(
        self,
        code: str,
        *,
        input_csv: Path,
        params: dict,
        exec_dir: Path,
        timeout_s: int | None = None,
    ) -> SandboxResult:
        """Run ``code`` with ``input_csv`` as INPUT_PATH; outputs land in ``exec_dir/out``."""
        import json

        timeout_s = timeout_s or self.settings.sandbox_timeout_s
        exec_dir = Path(exec_dir).resolve()  # subprocess cwd changes, so paths must be absolute
        input_csv = Path(input_csv).resolve()
        in_dir, out_dir = exec_dir / "in", exec_dir / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_csv, in_dir / "input.csv")
        (in_dir / "params.json").write_text(json.dumps(params), encoding="utf-8")
        (in_dir / "agent_code.py").write_text(code, encoding="utf-8")
        runner_src = resources.files("autoinsight").joinpath("scripts/runner.py").read_text("utf-8")
        (in_dir / "runner.py").write_text(runner_src, encoding="utf-8")

        started = time.monotonic()
        if self.mode == "docker":
            result = self._run_docker(in_dir, out_dir, timeout_s)
        else:
            result = self._run_local(in_dir, out_dir, timeout_s)
        result.duration_s = time.monotonic() - started
        removed, failed = sanitize_output(out_dir)
        if removed:
            result.stderr += f"\nRemoved non-regular files from the output directory: {removed[:5]}"
        if failed:
            result.ok = False
            result.stderr += f"\nThe program left links or special files that could not be removed: {failed[:5]}. Output rejected."
        result.files = sorted(
            str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file() and ".mpl" not in p.parts
        )
        if result.ok and _dir_size(out_dir) > MAX_OUT_BYTES:
            result.ok = False
            result.stderr += "\nOutput directory exceeded the size limit."
        return result

    # -- backends --------------------------------------------------------------------------------
    def _host_path(self, path: Path) -> str:
        """Translate a path when this process itself runs in a container that talks to the host
        Docker daemon (see docs/DEPLOYMENT.md, ``AUTOINSIGHT_HOST_RUNS_DIR``)."""
        host_runs = os.environ.get("AUTOINSIGHT_HOST_RUNS_DIR")
        if not host_runs:
            return str(path.resolve())
        runs = self.settings.runs_dir.resolve()
        return str(Path(host_runs) / path.resolve().relative_to(runs))

    def _run_docker(self, in_dir: Path, out_dir: Path, timeout_s: int) -> SandboxResult:
        os.chmod(out_dir, 0o777)  # the container runs as an unprivileged uid
        (out_dir / "charts").mkdir(exist_ok=True)  # created by the host so the host can always clean it up
        os.chmod(out_dir / "charts", 0o777)
        name = f"autoinsight-{uuid.uuid4().hex[:12]}"
        cmd = build_docker_command(
            image=self.settings.sandbox_image,
            in_dir=self._host_path(in_dir),
            out_dir=self._host_path(out_dir),
            name=name,
            memory_mb=self.settings.sandbox_memory_mb,
        )
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s + 15)
            timed_out = False
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True, check=False)
            proc.kill()
            stdout, stderr = proc.communicate()
            timed_out = True
        return SandboxResult(
            ok=(proc.returncode == 0 and not timed_out), returncode=proc.returncode,
            stdout=_trim(stdout or ""), stderr=_trim(stderr or ""), timed_out=timed_out,
            duration_s=0.0, mode="docker", out_dir=out_dir,
        )  # fmt: skip

    def _run_local(self, in_dir: Path, out_dir: Path, timeout_s: int) -> SandboxResult:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "AI_IN_DIR": str(in_dir), "AI_OUT_DIR": str(out_dir),
            "AI_LIMIT_MB": str(max(self.settings.sandbox_memory_mb, 3072)),
            "AI_LIMIT_CPU_S": str(timeout_s),
            "MPLCONFIGDIR": str(out_dir / ".mpl"), "HOME": str(out_dir),
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
        }  # fmt: skip
        proc = subprocess.Popen(
            [sys.executable, "-I", str(in_dir / "runner.py"), "agent_code.py"],
            cwd=out_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )  # fmt: skip
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            stdout, stderr = proc.communicate()
            timed_out = True
        if proc.returncode == -signal.SIGXCPU:  # RLIMIT_CPU hit before the wall-clock timeout
            timed_out = True
        return SandboxResult(
            ok=(proc.returncode == 0 and not timed_out), returncode=proc.returncode,
            stdout=_trim(stdout or ""), stderr=_trim(stderr or ""), timed_out=timed_out,
            duration_s=0.0, mode="local", out_dir=out_dir,
        )  # fmt: skip


def sanitize_output(out_dir: Path) -> tuple[list[str], list[str]]:
    """Delete everything in ``out_dir`` that is not a plain file or directory.

    Code in the sandbox controls what it writes to the output directory. A symlink there would point at
    a path that is resolved on the HOST when we read or copy it afterwards (``/etc/passwd``, ``~/.ssh``...),
    turning the output directory into a file-exfiltration channel. Symlinks, devices, sockets and FIFOs are
    removed before anything on the host reads the directory.

    Returns ``(removed, failed)`` - relative paths that were deleted, and ones that could not be deleted
    (the caller must then treat the run as failed rather than read the directory).
    """
    removed: list[str] = []
    failed: list[str] = []
    for root, dirs, files in os.walk(out_dir, topdown=True, followlinks=False):
        for name in list(dirs) + files:
            path = Path(root) / name
            if not (path.is_symlink() or not (path.is_dir() or path.is_file())):
                continue
            rel = str(path.relative_to(out_dir))
            try:
                path.unlink()
                removed.append(rel)
            except OSError:
                failed.append(rel)
            if name in dirs:
                dirs.remove(name)  # never descend into a link
    return removed, failed


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())