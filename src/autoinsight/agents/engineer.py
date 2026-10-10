"""Agent 1 - Data Engineer: writes and executes pandas code that cleans the raw CSV."""

from __future__ import annotations

from pathlib import Path

from .. import templates
from ..profiling import load_dataframe, profile_dataframe
from ..prompts import engineer_prompts
from ..state import AgentState, RunContext
from .code_agent import CodeAgentSpec, run_code_agent_step
from .contracts import validate_engineer


def _prompt(ctx: RunContext):
    def build(state: AgentState, previous_code, error):
        return engineer_prompts(state["question"], state["profile"], state["plan"], previous_code, error)

    return build


def make_spec(ctx: RunContext) -> CodeAgentSpec:
    raw_rows = int(load_dataframe(ctx.input_csv).shape[0])

    def validate(out_dir: Path, state: AgentState, _ctx: RunContext) -> list[str]:
        return validate_engineer(out_dir, state["plan"], raw_rows)

    return CodeAgentSpec(
        name="engineer",
        template=templates.ENGINEER,
        build_prompt=_prompt(ctx),
        input_csv=lambda c: c.input_csv,
        validate=validate,
    )


def engineer_node(state: AgentState, config) -> dict:
    ctx: RunContext = config["configurable"]["ctx"]
    return run_code_agent_step(ctx, state, make_spec(ctx))


def profile_cleaned(ctx: RunContext) -> dict:
    df = load_dataframe(ctx.artifacts / "cleaned.csv")
    return profile_dataframe(df, send_samples=ctx.settings.send_sample_values)
