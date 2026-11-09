"""Groq rate limits: short ones are retried by the client; ones it cannot wait out get a plain message."""
from fakes import FakeLLM

from parakh.agents.llm import friendly_error, get_llm
from parakh.agents.service import CaseService
from parakh.config import Settings
from parakh.patterns.library import PatternLibrary
from parakh.storage.cases import MemoryRepository

DAILY = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization "
         "`org_x` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199731, Requested 1450. "
         "Please try again in 1h12m3.5s. Need more tokens?', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}")
MINUTE = ("Error code: 429 - {'error': {'message': 'Rate limit reached ... on tokens per minute (TPM): Limit 8000, "
          "Used 7900, Requested 1200. Please try again in 8.25s.'}}")


class RateLimitError(Exception):  # same name and status code as groq.RateLimitError
    status_code = 429


def test_daily_and_minute_limits_read_plainly():
    assert friendly_error(RateLimitError(DAILY)) == (
        "rate_limit", "Groq's free daily limit for this model is used up. Try again in about 1 h 12 min.")
    assert friendly_error(RateLimitError(MINUTE)) == (
        "rate_limit", "Groq is busy (too many requests right now). Try again in about 9 s.")
    kind, message = friendly_error(ValueError("bad input"))
    assert kind == "error" and message == "ValueError: bad input"


def test_the_groq_client_retries_short_limits_itself(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")  # building the client makes no request
    llm = get_llm(Settings(llm_provider="groq", groq_model="openai/gpt-oss-120b"))
    assert llm.max_retries >= 5  # waits Groq's suggested time (up to 60 s) between tries


class _OutOfTokens(FakeLLM):
    """Out of daily tokens until `quota_back` is set, then a normal scripted model."""
    quota_back = False

    def invoke(self, msgs, config=None):
        if not self.quota_back:
            raise RateLimitError(DAILY)
        return super().invoke(msgs, config)


def test_a_case_stopped_by_the_daily_limit_says_so_and_can_be_retried(store):
    llm = _OutOfTokens()
    svc = CaseService(cfg=Settings(), store=store, repo=MemoryRepository(), llm=llm, library=PatternLibrary())
    case = svc.repo.get("cases", svc.open_case(svc.alerts[0]["account"], background=False))
    assert case["status"] == "failed" and case["error_kind"] == "rate_limit"
    assert case["error"].startswith("Groq's free daily limit")
    llm.quota_back = True  # the next day
    retried = svc.retry(case["_id"], background=False)
    assert retried["status"] != "failed" and retried["error_kind"] is None
