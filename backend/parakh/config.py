"""Central configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BACKEND_DIR / "data" / "demo_transactions.csv.gz"
load_dotenv(BACKEND_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0")))
    # Price in USD per 1M tokens, used only to estimate cost per case.
    price_input_per_m: float = field(default_factory=lambda: float(os.getenv("PRICE_INPUT_PER_M", "0.15")))
    price_output_per_m: float = field(default_factory=lambda: float(os.getenv("PRICE_OUTPUT_PER_M", "0.60")))

    # Agent budgets
    max_tool_calls: int = field(default_factory=lambda: int(os.getenv("MAX_TOOL_CALLS", "8")))
    max_report_retries: int = field(default_factory=lambda: int(os.getenv("MAX_REPORT_RETRIES", "2")))
    max_cases_per_run: int = field(default_factory=lambda: int(os.getenv("MAX_CASES_PER_RUN", "10")))

    # Data and storage
    data_path: str = field(default_factory=lambda: os.getenv("DATA_PATH", str(DEFAULT_DATA_PATH)))
    mongodb_uri: str | None = field(default_factory=lambda: os.getenv("MONGODB_URI") or None)
    mongodb_db: str = field(default_factory=lambda: os.getenv("MONGODB_DB", "parakh"))

    # API
    cors_origins: list[str] = field(
        default_factory=lambda: [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
    )
    verifier_enabled: bool = field(default_factory=lambda: _bool("VERIFIER_ENABLED", True))


settings = Settings()
