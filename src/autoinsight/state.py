"""Shared state that flows through the LangGraph workflow, plus the per-run context."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from .config import Settings
from .llm import LLMClient
from .sandbox import Sandbox

STAGES = ["profile", "planner", "engineer", "analyst", "modeler", "reporter"]


class AgentStatus(TypedDict, total=False):
    attempts: int
    status: str  # pending | ok | retry | failed | skipped
    last_error: str | None
    last_code: str | None
    used_fallback: bool


class AgentState(TypedDict, total=False):
    question: str
    target_hint: str | None
    profile: dict
    plan: dict
    agents: dict[str, AgentStatus]
    facts: dict
    narrative: str
    grounding: dict
    model_error: str | None
    error: str | None  # set => workflow routes to the failure node
    outcome: str  # completed | failed
    events: Annotated[list[dict], operator.add]


@dataclass
class RunContext:
    """Everything a node needs that is not (and should not be) part of the serialisable state."""

    settings: Settings
    llm: LLMClient
    sandbox: Sandbox
    run_dir: Path
    emit: Callable[[dict], None] = field(default=lambda event: None)

    @property
    def input_csv(self) -> Path:
        return self.run_dir / "input.csv"

    @property
    def artifacts(self) -> Path:
        path = self.run_dir / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs(self) -> Path:
        path = self.run_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def work_dir(self, agent: str, attempt: int) -> Path:
        return self.run_dir / "work" / f"{agent}_attempt{attempt}"

    def sandbox_params(self, plan: dict) -> dict[str, Any]:
        return {**plan, "seed": self.settings.seed}
