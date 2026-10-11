"""Trusted entry point that runs INSIDE the sandbox (Docker container or isolated subprocess).

It builds the namespace that agent code can rely on, then executes the agent's program:

    INPUT_PATH            path of the CSV to analyse (read-only)
    OUT_DIR               directory where outputs must be written
    PARAMS                dict with the analysis plan (target, task_type, ...)
    save_csv(df, name)    write a DataFrame to OUT_DIR/name
    save_json(obj, name)  write JSON (NaN/Inf -> null, numpy types handled) to OUT_DIR/name
    save_fig(fig, name)   write a matplotlib figure to OUT_DIR/charts/name (PNG)
    log(*args)            print to stdout (captured and shown to the agents)

Usage: ``python -I runner.py <code-file-inside-IN_DIR>``
"""

import json
import math
import os
import re
import sys
import traceback
import warnings

IN_DIR = os.environ["AI_IN_DIR"]
OUT_DIR = os.environ["AI_OUT_DIR"]
CODE_NAME = sys.argv[1]


def _apply_local_limits():
    """Resource limits for local (non-Docker) mode. Docker enforces its own limits."""
    try:
        import resource
    except ImportError:  # Windows
        return
    mem_mb = int(os.environ.get("AI_LIMIT_MB", "0") or 0)
    cpu_s = int(os.environ.get("AI_LIMIT_CPU_S", "0") or 0)
    if mem_mb:
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
    if cpu_s:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
    if os.environ.get("AI_LIMIT_MB"):
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 * 1024, 256 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (512, 512))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


_apply_local_limits()
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

os.makedirs(os.path.join(OUT_DIR, "charts"), exist_ok=True)


def _clean(obj):
    """Make anything JSON-safe: numpy scalars -> python, NaN/Inf -> None, sets -> lists."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def _safe_name(name, suffix=None):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(str(name))) or "output"
    if suffix and not name.lower().endswith(suffix):
        name += suffix
    return name


def save_json(obj, name):
    with open(os.path.join(OUT_DIR, _safe_name(name, ".json")), "w", encoding="utf-8") as fh:
        json.dump(_clean(obj), fh, indent=2)


def save_csv(df, name):
    df.to_csv(os.path.join(OUT_DIR, _safe_name(name, ".csv")), index=False)


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, "charts", _safe_name(name, ".png"))
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def log(*args):
    print(*args, flush=True)


def _format_traceback(source_lines):
    """Show only the frames from the agent's code + the exception (runner frames are noise)."""
    etype, evalue, tb = sys.exc_info()
    frames = [f for f in traceback.extract_tb(tb) if f.filename == CODE_NAME]
    lines = ["Traceback (most recent call last):"]
    for f in frames:
        text = source_lines[f.lineno - 1].strip() if f.lineno and f.lineno <= len(source_lines) else ""
        lines.append(f"  line {f.lineno}, in {f.name}: {text}")
    lines.extend(traceback.format_exception_only(etype, evalue))
    return "\n".join(line.rstrip("\n") for line in lines)


def _open_permissions():
    """Make everything in OUT_DIR readable and replaceable by the host user.

    In Docker mode this process runs as an unprivileged uid that differs from the host user's; whatever it
    creates is owned by it. The directory is a throw-away per-run folder, so world-accessible modes are fine.
    """
    for root, dirs, files in os.walk(OUT_DIR):
        for name in dirs + files:
            path = os.path.join(root, name)
            if os.path.islink(path):
                continue
            try:
                os.chmod(path, 0o777 if os.path.isdir(path) else 0o666)
            except OSError:
                pass


def main():
    in_container = os.path.exists("/.dockerenv")  # local mode runs as the host user: leave its modes alone
    if in_container:
        os.umask(0)  # new files 0666 / dirs 0777: the host must be able to read and replace what we write
    try:
        _run()
    finally:
        if in_container:
            _open_permissions()


def _run():
    with open(os.path.join(IN_DIR, "params.json"), encoding="utf-8") as fh:
        params = json.load(fh)
    with open(os.path.join(IN_DIR, CODE_NAME), encoding="utf-8") as fh:
        source = fh.read()

    namespace = {
        "__name__": "__main__",
        "INPUT_PATH": os.path.join(IN_DIR, "input.csv"),
        "OUT_DIR": OUT_DIR,
        "PARAMS": params,
        "save_json": save_json,
        "save_csv": save_csv,
        "save_fig": save_fig,
        "log": log,
        "plt": plt,
    }
    try:
        exec(compile(source, CODE_NAME, "exec"), namespace)  # noqa: S102 - the whole point
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - report everything back to the agent
        print(_format_traceback(source.splitlines()), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()