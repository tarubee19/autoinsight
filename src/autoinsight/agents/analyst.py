"""Agent 2 - Analyst: statistical tests + charts that explain what drives the target."""

from __future__ import annotations

from pathlib import Path

from .. import templates
from ..prompts import analyst_prompts
from ..state import AgentState, RunContext
from .code_agent import CodeAgentSpec, run_code_agent_step
from .contracts import normalise_stats, validate_analyst
from .engineer import profile_cleaned


def make_spec(ctx: RunContext, cleaned_profile: dict) -> CodeAgentSpec:
    def build(state: AgentState, previous_code, error):
        return analyst_prompts(state["question"], state["plan"], cleaned_profile, previous_code, error)

    def validate(out_dir: Path, state: AgentState, c: RunContext) -> list[str]:
        return validate_analyst(out_dir, state["plan"], c.artifacts / "cleaned.csv")

    return CodeAgentSpec(
        name="analyst",
        template=templates.ANALYST,
        build_prompt=build,
        input_csv=lambda c: c.artifacts / "cleaned.csv",
        validate=validate,
        post_process=normalise_stats,
    )


def analyst_node(state: AgentState, config) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    return run_code_agent_step(ctx, state, make_spec(ctx, profile_cleaned(ctx)))
