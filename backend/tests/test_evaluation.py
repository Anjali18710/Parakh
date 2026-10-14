from parakh.eval.run_eval import pick_sample, screening_coverage, summarise


def _row(mode, acc, bad, verdict, pattern="fan_in", truth=("fan_in",), role="retail customer", **kw):
    return {"mode": mode, "account": acc, "truly_laundering": bad, "true_patterns": list(truth) if bad else [],
            "role": role, "verdict": verdict, "pattern_id": pattern, "risk_score": 80, "tokens": 1000,
            "cost_usd": 0.001, "seconds": 10, "claims_total": 2, "claims_verified": 2, **kw}


def test_summary_metrics():
    rows = [_row("full", "A", True, "suspicious"),                       # caught, right pattern
            _row("full", "B", True, "suspicious", pattern="cycle"),      # caught, wrong pattern
            _row("full", "C", True, "not_suspicious"),                   # missed
            _row("full", "D", False, "suspicious", role="distributor"),  # false alarm
            _row("full", "E", False, "not_suspicious")]                  # cleared
    meta = {"name": "t", "modes": ["full"], "sample_size": 5}
    m = summarise(rows, meta)["results"]["full"]
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == round(2 / 3, 3) and m["recall"] == round(2 / 3, 3)
    assert m["pattern_accuracy"] == 0.5 and m["innocent_cleared"] == 0.5
    assert m["false_alarms_by_role"] == {"distributor": 1} and m["missed_by_pattern"] == {"fan_in": 1}


def test_errors_are_counted_not_scored():
    rows = [_row("full", "A", True, "suspicious"), {**_row("full", "B", True, None), "error": "boom"}]
    m = summarise(rows, {"name": "t", "modes": ["full"], "sample_size": 2})["results"]["full"]
    assert m["cases"] == 1 and m["errors"] == 1 and m["recall"] == 1.0


def test_sample_is_balanced_and_repeatable():
    alerts = [{"account": f"A{i}"} for i in range(40)]
    truth = {f"A{i}": {"x"} for i in range(0, 40, 3)}
    s1, s2 = pick_sample(alerts, truth, 10, 7), pick_sample(alerts, truth, 10, 7)
    assert s1 == s2 and len(s1) == 10 and sum(a in truth for a in s1) == 5


def test_external_sources_excluded_from_coverage():
    cov = screening_coverage({"A": {}}, {"A": {"x"}, "B": {"x"}, "EXT": {"x"}}, {"EXT": "external source"})
    assert cov == {"alerts": 1, "laundering_accounts": 2, "laundering_accounts_alerted": 1}
