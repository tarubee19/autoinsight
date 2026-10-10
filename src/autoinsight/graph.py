"""The LangGraph workflow.

        +---------+    +---------+    +----------+    +---------+    +---------+    +----------+
 START->| profile |--->| planner |--->| engineer |--->| analyst |--->| modeler |--->| reporter |--> finalize -> END
        +---------+    +---------+    +-----^----+    +----^----+    +---------+    +----------+
              \\             \\          retry|  (self-loop)|retry          (soft-fails)
               \\_____________\\__________  failure  ______/
                                           v
                                         fail -> END

* ``engineer`` and ``analyst`` are *cycles*: when their code fails the safety guard, the sandbox
  or the output contract, the same node runs again with the error as feedback.
* Any unrecoverable error routes to ``fail`` which still writes a readable report + summary.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents.analyst import analyst_node
from .agents.engineer import engineer_node
from .agents.modeler import modeler_node
from .agents.planner import planner_node
from .agents.reporter import reporter_node
from .agents.terminal import fail_node, finalize_node, profile_node
from .state import AgentState


def _after(next_node: str, *, retry_node: str | None = None):
    """Router for a node: failure -> 'fail', retry -> same node, otherwise continue."""

    def route(state: AgentState) -> str:
        if state.get("error"):
            return "fail"
        if retry_node:
            if state.get("agents", {}).get(retry_node, {}).get("status") == "retry":
                return retry_node
        return next_node

    return route


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("profile", profile_node)
    g.add_node("planner", planner_node)
    g.add_node("engineer", engineer_node)
    g.add_node("analyst", analyst_node)
    g.add_node("modeler", modeler_node)
    g.add_node("reporter", reporter_node)
    g.add_node("finalize", finalize_node)
    g.add_node("fail", fail_node)

    g.add_edge(START, "profile")
    g.add_conditional_edges("profile", _after("planner"), {"planner": "planner", "fail": "fail"})
    g.add_conditional_edges("planner", _after("engineer"), {"engineer": "engineer", "fail": "fail"})
    g.add_conditional_edges("engineer", _after("analyst", retry_node="engineer"), {"analyst": "analyst", "engineer": "engineer", "fail": "fail"})
    g.add_conditional_edges("analyst", _after("modeler", retry_node="analyst"), {"modeler": "modeler", "analyst": "analyst", "fail": "fail"})
    g.add_edge("modeler", "reporter")
    g.add_conditional_edges("reporter", _after("finalize"), {"finalize": "finalize", "fail": "fail"})
    g.add_edge("finalize", END)
    g.add_edge("fail", END)
    return g.compile()
