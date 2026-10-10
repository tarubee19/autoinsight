"""Test doubles shipped with the package so tests, the evaluation script and users can simulate
an unreliable LLM (bad code, outages, hallucinated numbers) without any API key."""

from __future__ import annotations

from . import templates
from .llm import LLMRequest, OfflineClient


class ScriptedLLM:
    """An 'LLM' whose answers per agent role are scripted.

    ``ScriptedLLM(engineer=[bad_code, other_bad_code, good_code])`` returns those answers in order.
    An ``Exception`` instance in the list is raised (simulating an outage); a callable is called with
    the request. Once a role's script is exhausted, the offline (template) answer is returned.
    Every request is recorded in ``.requests`` so tests can inspect the prompts the agents sent.
    """

    is_offline = False
    name = "scripted:test"

    def __init__(self, **scripts):
        self.scripts = {role: list(items) for role, items in scripts.items()}
        self.calls: dict[str, int] = {}
        self.requests: list[LLMRequest] = []
        self._offline = OfflineClient()

    def complete(self, req: LLMRequest) -> str:
        self.requests.append(req)
        self.calls[req.role] = self.calls.get(req.role, 0) + 1
        queue = self.scripts.get(req.role)
        if queue:
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item(req) if callable(item) else item
        return self._offline.complete(req)


def fenced(code: str) -> str:
    return f"Here is the program:\n```python\n{code}\n```"


def engineer_template(_req=None) -> str:
    return fenced(templates.source(templates.ENGINEER))


def analyst_template(_req=None) -> str:
    return fenced(templates.source(templates.ANALYST))


# Canned failure modes -------------------------------------------------------------------------
UNSAFE_CODE = fenced("import os\nos.system('echo pwned > /tmp/pwned')")
RUNTIME_BUG = fenced("import pandas as pd\ndf = pd.read_csv(INPUT_PATH)\ndf['x'] = df['NoSuchColumn'] * 2\nsave_csv(df, 'cleaned.csv')")
NO_OUTPUT = fenced("import pandas as pd\ndf = pd.read_csv(INPUT_PATH)\nlog(len(df))")
HALLUCINATED_FEATURE = fenced("save_json({'top_drivers': [{'feature': 'Imaginary', 'strength': 0.9, 'finding': 'x'}], 'base_rate': 0.3}, 'stats.json')")
INVENTED_STATS_REPORT = "## Executive summary\n- Churn is exactly 87.65% and revenue fell 42.1%.\n"
