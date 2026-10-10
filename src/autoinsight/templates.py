"""Access to the vetted sandbox programs shipped inside the package."""

from __future__ import annotations

from importlib import resources

ENGINEER = "engineer_template.py"
ANALYST = "analyst_template.py"
TRAINER = "train_model.py"


def source(name: str) -> str:
    return resources.files("autoinsight").joinpath(f"scripts/{name}").read_text(encoding="utf-8")
