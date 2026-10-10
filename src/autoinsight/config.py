"""Runtime configuration, read from environment variables (prefix ``AUTOINSIGHT_``).

Everything has a safe default so ``autoinsight run data.csv`` works with zero setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROVIDERS = ("auto", "anthropic", "openai", "offline")
SANDBOX_MODES = ("auto", "docker", "local")


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"AUTOINSIGHT_{name}", default)


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "auto"  # auto | anthropic | openai | offline
    llm_model: str | None = None  # provider default if None
    sandbox_mode: str = "auto"  # auto | docker | local
    sandbox_image: str = "autoinsight-sandbox:latest"
    sandbox_timeout_s: int = 120
    sandbox_memory_mb: int = 2048
    max_code_attempts: int = 3  # LLM attempts per code-writing agent before the vetted fallback
    max_upload_mb: int = 50
    runs_dir: Path = Path("runs")
    allow_unsafe_local: bool = False  # allow LLM-written code in the *local* (non-Docker) sandbox
    send_sample_values: bool = True  # include 3 example values per column in LLM prompts
    seed: int = 42

    @classmethod
    def from_env(cls) -> Settings:
        provider = (_env("LLM_PROVIDER", "auto") or "auto").lower()
        if provider not in PROVIDERS:
            raise ValueError(f"AUTOINSIGHT_LLM_PROVIDER must be one of {PROVIDERS}, got {provider!r}")
        mode = (_env("SANDBOX_MODE", "auto") or "auto").lower()
        if mode not in SANDBOX_MODES:
            raise ValueError(f"AUTOINSIGHT_SANDBOX_MODE must be one of {SANDBOX_MODES}, got {mode!r}")
        return cls(
            llm_provider=provider,
            llm_model=_env("MODEL"),
            sandbox_mode=mode,
            sandbox_image=_env("SANDBOX_IMAGE", "autoinsight-sandbox:latest") or "autoinsight-sandbox:latest",
            sandbox_timeout_s=int(_env("SANDBOX_TIMEOUT_S", "120") or 120),
            sandbox_memory_mb=int(_env("SANDBOX_MEMORY_MB", "2048") or 2048),
            max_code_attempts=int(_env("MAX_CODE_ATTEMPTS", "3") or 3),
            max_upload_mb=int(_env("MAX_UPLOAD_MB", "50") or 50),
            runs_dir=Path(_env("RUNS_DIR", "runs") or "runs"),
            allow_unsafe_local=_bool(_env("ALLOW_UNSAFE_LOCAL"), False),
            send_sample_values=_bool(_env("SEND_SAMPLE_VALUES"), True),
            seed=int(_env("SEED", "42") or 42),
        )

    def resolved_provider(self) -> str:
        """Turn ``auto`` into a concrete provider based on which API keys exist."""
        if self.llm_provider != "auto":
            return self.llm_provider
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return "offline"
