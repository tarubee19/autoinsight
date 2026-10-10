"""LLM abstraction layer.

Every agent talks to an :class:`LLMClient`. Three implementations exist:

* :class:`AnthropicClient` - Claude via the ``anthropic`` SDK (``ANTHROPIC_API_KEY``)
* :class:`OpenAIClient`    - GPT via the ``openai`` SDK (``OPENAI_API_KEY``)
* :class:`OfflineClient`   - no network, no key. Instead of *writing* code it fills in the vetted
  templates in ``autoinsight/scripts``. This keeps the whole graph runnable in CI, in demos and for
  anyone without an API key, and doubles as the deterministic fallback when an LLM keeps failing.

The :class:`LLMRequest` carries both the prompt text (used by real LLMs) and a structured
``context`` dict (used only by the offline client).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o",
}


@dataclass
class LLMRequest:
    role: str  # planner | engineer | analyst | reporter
    system: str
    user: str
    context: dict = field(default_factory=dict)
    json_mode: bool = False
    max_tokens: int = 4096


class LLMClient(Protocol):
    name: str
    is_offline: bool

    def complete(self, req: LLMRequest) -> str: ...


class AnthropicClient:
    is_offline = False

    def __init__(self, model: str | None = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError("Install the LLM extra: pip install 'autoinsight[llm]'") from exc
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.name = f"anthropic:{self.model}"
        self._client = anthropic.Anthropic(max_retries=4, timeout=180.0)

    def complete(self, req: LLMRequest) -> str:  # pragma: no cover - needs network + key
        kwargs = {}
        if os.environ.get("AUTOINSIGHT_TEMPERATURE"):
            kwargs["temperature"] = float(os.environ["AUTOINSIGHT_TEMPERATURE"])
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=req.max_tokens,
            system=req.system,
            messages=[{"role": "user", "content": req.user}],
            **kwargs,
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


class OpenAIClient:
    is_offline = False

    def __init__(self, model: str | None = None):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError("Install the LLM extra: pip install 'autoinsight[llm]'") from exc
        self.model = model or DEFAULT_MODELS["openai"]
        self.name = f"openai:{self.model}"
        self._client = openai.OpenAI(max_retries=4, timeout=180.0)

    def complete(self, req: LLMRequest) -> str:  # pragma: no cover - needs network + key
        kwargs = {}
        if req.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if os.environ.get("AUTOINSIGHT_TEMPERATURE"):
            kwargs["temperature"] = float(os.environ["AUTOINSIGHT_TEMPERATURE"])
        resp = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=req.max_tokens,
            messages=[
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content or ""


class OfflineClient:
    """Deterministic 'LLM' that answers from templates; see :mod:`autoinsight.offline`."""

    is_offline = True
    name = "offline:templates"

    def complete(self, req: LLMRequest) -> str:
        from . import offline

        return offline.respond(req)


def build_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or Settings.from_env()
    provider = settings.resolved_provider()
    if provider == "anthropic":
        return AnthropicClient(settings.llm_model)
    if provider == "openai":
        return OpenAIClient(settings.llm_model)
    return OfflineClient()


# ---------------------------------------------------------------------------
# Output parsing helpers (LLMs wrap things in prose and markdown fences)
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Return the python code from an LLM answer (fenced block if present, else the raw text)."""
    blocks = _FENCE.findall(text)
    if blocks:
        # The longest block is almost always the real program, not a usage snippet.
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"


def extract_json(text: str) -> dict:
    """Parse the first JSON object in ``text``; raises ``ValueError`` if none is valid."""
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for cand in candidates:
        start = cand.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(cand[start : i + 1])
                        except json.JSONDecodeError:
                            break
                        if isinstance(obj, dict):
                            return obj
                        break
            start = cand.find("{", start + 1)
    raise ValueError("No valid JSON object found in LLM output")
