"""LLM factory and token-usage tracking."""
from __future__ import annotations

import re
import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from parakh.config import Settings, settings as default_settings


def get_llm(cfg: Settings | None = None):
    cfg = cfg or default_settings
    if cfg.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=cfg.gemini_model, temperature=cfg.temperature)
    from langchain_groq import ChatGroq

    extra = {}
    if "gpt-oss" in cfg.groq_model:
        # Reasoning tokens count against the rate limit; "low" is plenty for tool-driven investigation.
        extra["reasoning_effort"] = cfg.reasoning_effort
    return ChatGroq(model=cfg.groq_model, temperature=cfg.temperature, max_retries=6, **extra)


_TRY_AGAIN = re.compile(r"try again in\s+([0-9hms.]+)", re.I)


def friendly_error(exc: BaseException) -> tuple[str, str]:
    """(kind, message) for an error that stopped a case. Groq's rate limits get a plain explanation.

    Short waits (requests or tokens per minute) are already retried by the Groq client. What reaches
    here is a limit it could not wait out, usually the free tier's daily token limit.
    """
    text = str(exc)
    is_limit = getattr(exc, "status_code", None) == 429 or type(exc).__name__ == "RateLimitError"
    if not is_limit:
        return "error", f"{type(exc).__name__}: {text}"
    wait = _TRY_AGAIN.search(text)
    when = f" Try again in about {_readable(wait.group(1))}." if wait else " Try again later."
    if "per day" in text.lower() or "(tpd)" in text.lower() or "(rpd)" in text.lower():
        return "rate_limit", "Groq's free daily limit for this model is used up." + when
    return "rate_limit", "Groq is busy (too many requests right now)." + when


def _readable(span: str) -> str:
    """'1h2m3.5s' -> '1 h 2 min', '45.2s' -> '46 s'."""
    h, m, s = (re.search(rf"([0-9.]+){u}", span) for u in ("h", "m(?!s)", "s"))
    hours, mins, secs = (float(x.group(1)) if x else 0.0 for x in (h, m, s))
    if hours:
        return f"{int(hours)} h {int(mins)} min" if mins else f"{int(hours)} h"
    if mins:
        return f"{int(mins)} min"
    return f"{int(secs + 0.999)} s"


class UsageTracker(BaseCallbackHandler):
    """Counts tokens and LLM calls so we can report cost per case."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self._lock = threading.Lock()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        with self._lock:
            self.calls += 1
            for gen_list in response.generations:
                for gen in gen_list:
                    meta = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                    self.input_tokens += int(meta.get("input_tokens", 0))
                    self.output_tokens += int(meta.get("output_tokens", 0))

    def summary(self, cfg: Settings | None = None) -> dict:
        cfg = cfg or default_settings
        cost = (self.input_tokens * cfg.price_input_per_m + self.output_tokens * cfg.price_output_per_m) / 1e6
        return {"llm_calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "est_cost_usd": round(cost, 5)}
