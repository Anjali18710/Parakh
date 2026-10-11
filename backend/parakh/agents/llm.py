"""LLM factory and token-usage tracking."""
from __future__ import annotations

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

    return ChatGroq(model=cfg.groq_model, temperature=cfg.temperature, max_retries=4)


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
