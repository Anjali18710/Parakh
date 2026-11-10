"""The rules-vs-Parakh comparison: fair sample, resumes where it stopped, stops cleanly on the daily limit."""
import json

import pytest
from fakes import FakeLLM
from test_rate_limits import DAILY, RateLimitError

from parakh.eval.compare import DailyLimit, markdown, pick_sample, run, summarise, true_patterns
from parakh.tools.screening import screen


class _RunsOutAfter(FakeLLM):
    """A model whose daily quota runs out after `cases` investigations."""

    def __init__(self, cases: int):
        super().__init__()
        self.left = cases

    def with_structured_output(self, schema):
        inner = super().with_structured_output(schema)
        parent = self

        class _Counting:
            def invoke(self, msgs, config=None):
                if schema.__name__ == "CaseReport" and parent.left <= 0:
                    raise RateLimitError(DAILY)
                result = inner.invoke(msgs, config)
                if schema.__name__ == "CaseReport":
                    parent.left -= 1
                return result
        return _Counting()


def test_sample_is_half_laundering_and_the_same_every_time(store):
    alerts = [a.to_dict() for a in screen(store)]
    laundering = set(true_patterns(store))
    sample = pick_sample(alerts, laundering, 10, seed=7)
    assert len(sample) == 10 and len(set(sample)) == 10
    assert sum(acc in laundering for acc in sample) == 5
    assert sample == pick_sample(alerts, laundering, 10, seed=7)


def test_stops_on_the_daily_limit_and_carries_on_next_time(store, tmp_path):
    alerts = [a.to_dict() for a in screen(store)]
    with pytest.raises(DailyLimit, match="daily limit"):
        run(store, alerts, _RunsOutAfter(cases=2), tmp_path, n=6, log=lambda *_: None)
    saved = (tmp_path / "cases.jsonl").read_text().splitlines()
    assert len(saved) >= 2  # finished cases are kept (a rewrite may use up a report call)

    rows = run(store, alerts, FakeLLM(), tmp_path, n=6, log=lambda *_: None)  # "the next day"
    sample = json.loads((tmp_path / "sample.json").read_text())["accounts"]
    assert [r["account"] for r in rows] == sample and all("parakh" in r for r in rows)

    s = summarise(rows, len(sample))
    assert s["rules"]["cases"] == s["parakh"]["cases"] == 6
    assert s["rules"]["laundering"] == 3 and s["rules"]["innocent"] == 3
    text = markdown(s, "fake-model")
    assert "| Rules alone (score >= 2) |" in text and "| Parakh |" in text and "so far" not in text


def test_detail_is_saved_and_the_dev_set_never_touches_the_test_set(store, tmp_path):
    alerts = [a.to_dict() for a in screen(store)]
    laundering = set(true_patterns(store))
    test_set = set(pick_sample(alerts, laundering, 6, seed=7))
    rows = run(store, alerts, FakeLLM(), tmp_path, n=4, seed=99, log=lambda *_: None, exclude=test_set)
    assert rows and not {r["account"] for r in rows} & test_set
    detail = rows[0]["parakh"]
    assert detail["summary"] and detail["claim_texts"] and detail["pattern_candidates"]


class _BadToolCallOnce(FakeLLM):
    """Like GPT-OSS on Groq now and then: the first reply calls a tool that does not exist."""
    failed = False

    def invoke(self, msgs, config=None):
        if not self.failed:
            self.failed = True
            raise ValueError("Error code: 400 - {'error': {'message': \"Tool call validation failed: attempted to call "
                             "tool 'json' which was not in request.tools\", 'code': 'tool_use_failed'}}")
        return super().invoke(msgs, config)


def test_a_made_up_tool_call_is_asked_again_instead_of_failing_the_case(store, tmp_path):
    alerts = [a.to_dict() for a in screen(store)]
    rows = run(store, alerts, _BadToolCallOnce(), tmp_path, n=2, log=lambda *_: None)
    assert all("parakh" in r and "error" not in r for r in rows)
