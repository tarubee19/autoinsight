"""Shared test doubles (thin aliases so tests read naturally)."""

from autoinsight import testing as _t

ScriptedLLM = _t.ScriptedLLM
fenced = _t.fenced
UNSAFE_CODE = _t.UNSAFE_CODE
RUNTIME_BUG = _t.RUNTIME_BUG
NO_OUTPUT = _t.NO_OUTPUT
HALLUCINATED_FEATURE = _t.HALLUCINATED_FEATURE
INVENTED_STATS_REPORT = _t.INVENTED_STATS_REPORT
ENGINEER_TEMPLATE = _t.engineer_template
ANALYST_TEMPLATE = _t.analyst_template
