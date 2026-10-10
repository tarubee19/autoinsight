"""Static safety checks for LLM-written code (first line of defence, *before* the sandbox).

The guard parses the code with :mod:`ast` and rejects anything outside a small, data-science-only
allow-list: no ``os``/``subprocess``/``socket`` imports, no ``eval``/``exec``/``open``, no dunder
tricks, no absolute paths or URLs. It is deliberately conservative.

IMPORTANT: an AST filter is a *best-effort pre-filter*, not a security boundary - Python is too
dynamic for that. The real isolation is the Docker sandbox (no network, read-only filesystem,
non-root, CPU/memory/pids limits). See ``docs/SECURITY.md``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

ALLOWED_IMPORTS = {
    "pandas", "numpy", "scipy", "matplotlib", "sklearn", "statsmodels", "seaborn",
    "math", "re", "json", "warnings", "itertools", "collections", "statistics",
    "datetime", "typing", "textwrap", "numbers", "functools", "string", "random",
}  # fmt: skip

FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input", "breakpoint", "exit", "quit",
    "getattr", "setattr", "delattr", "globals", "locals", "vars", "memoryview", "help",
}  # fmt: skip

# Attribute names that read/write arbitrary files, run code or hit the network via pandas/numpy/etc.
FORBIDDEN_ATTRS = {
    "read_pickle", "to_pickle", "read_clipboard", "to_clipboard", "read_sql", "read_sql_query",
    "read_sql_table", "to_sql", "read_html", "read_xml", "read_parquet", "read_feather",
    "read_hdf", "to_hdf", "read_orc", "read_spss", "read_sas", "read_stata", "read_gbq",
    "load", "loads", "dump", "dumps_pickle", "fromfile", "tofile", "memmap", "system", "popen",
    "Popen", "spawn", "fork", "urlopen", "urlretrieve", "ctypes", "frombuffer_unsafe",
}  # fmt: skip

_URL_OR_ABS_PATH = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*://|/|~|[A-Za-z]:[\\/])")


@dataclass
class GuardResult:
    ok: bool
    violations: list[str]

    def message(self) -> str:
        return "; ".join(self.violations)


def check_code(code: str) -> GuardResult:
    """Return ``GuardResult(ok=False, violations=[...])`` if the code breaks any rule."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return GuardResult(False, [f"SyntaxError: {exc.msg} (line {exc.lineno})"])

    violations: list[str] = []

    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")

        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    violations.append(f"line {line}: import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                violations.append(f"line {line}: import from '{node.module}' is not allowed")

        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                violations.append(f"line {line}: call to '{func.id}()' is not allowed")

        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                violations.append(f"line {line}: access to dunder attribute '{node.attr}' is not allowed")
            elif node.attr in FORBIDDEN_ATTRS:
                violations.append(f"line {line}: use of '.{node.attr}' is not allowed")

        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id != "__name__":
                violations.append(f"line {line}: use of name '{node.id}' is not allowed")

        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if len(value) < 300 and (_URL_OR_ABS_PATH.match(value) or ".." in value.split("/")):
                violations.append(
                    f"line {line}: string {value[:40]!r} looks like a URL or absolute path - "
                    "use INPUT_PATH/OUT_DIR helpers (save_csv, save_json, save_fig) only"
                )

        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append(f"line {line}: global/nonlocal statements are not allowed")

    # De-duplicate while keeping order
    seen: set[str] = set()
    unique = [v for v in violations if not (v in seen or seen.add(v))]
    return GuardResult(ok=not unique, violations=unique)
