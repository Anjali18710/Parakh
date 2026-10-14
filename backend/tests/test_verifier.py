from parakh.agents.verifier import _amounts_in_text, verify_report


def _txns(store, n=2):
    acc = store.df.from_account.value_counts().index[0]
    rows = store.out_txns(acc).head(n)
    return acc, rows


def test_correct_claims_pass(store):
    acc, rows = _txns(store)
    total = float(rows.amount.sum())
    report = {"verdict": "suspicious", "claims": [
        {"statement": f"{acc} sent ₹{total:,.2f} in two payments.", "txn_ids": list(rows.txn_id),
         "accounts": [acc], "amount": total, "count": 2},
        {"statement": f"{acc} made a payment.", "txn_ids": [rows.txn_id.iloc[0]], "accounts": [acc]},
    ]}
    result = verify_report(report, store, acc)
    assert result["passed"], result["issues"]
    assert result["claims_verified"] == 2


def test_invented_transaction_is_caught(store):
    acc, rows = _txns(store, 1)
    report = {"verdict": "not_suspicious", "claims": [
        {"statement": "A transfer happened.", "txn_ids": ["T9999999"], "accounts": []},
        {"statement": "Real one.", "txn_ids": [rows.txn_id.iloc[0]], "accounts": [acc]},
    ]}
    result = verify_report(report, store, acc)
    assert not result["passed"]
    assert "do not exist" in result["claim_results"][0]["problems"][0]


def test_wrong_amount_and_wrong_account_are_caught(store):
    acc, rows = _txns(store, 1)
    report = {"verdict": "not_suspicious", "claims": [
        {"statement": "Sent ₹99 lakh.", "txn_ids": [rows.txn_id.iloc[0]], "accounts": [acc, "AC000001"]},
    ]}
    problems = verify_report(report, store, acc)["claim_results"][0]["problems"]
    assert any("AC000001" in p for p in problems)
    assert any("does not match" in p for p in problems)


def test_suspicious_needs_two_verified_claims(store):
    acc, rows = _txns(store, 1)
    report = {"verdict": "suspicious", "claims": [
        {"statement": "One payment.", "txn_ids": [rows.txn_id.iloc[0]], "accounts": [acc]}]}
    assert not verify_report(report, store, acc)["passed"]


def test_amount_parsing_handles_indian_units_and_ignores_ids_and_dates():
    assert _amounts_in_text("moved ₹9.5 lakh and then ₹1.2 crore") == [950000.0, 12000000.0]
    assert _amounts_in_text("On 2026-07-09 14:25 AC123456 sent T0001234 in 2026") == []
    assert _amounts_in_text("₹66,41,114 left within 48h, 98% of it") == [6641114.0]


def test_threshold_amounts_are_not_treated_as_facts(store):
    acc, rows = _txns(store, 2)
    report = {"verdict": "suspicious", "claims": [
        {"statement": f"{acc} made two payments above ₹1,000.", "txn_ids": list(rows.txn_id), "accounts": [acc]},
        {"statement": f"{acc} made a payment.", "txn_ids": [rows.txn_id.iloc[0]], "accounts": [acc]},
    ]}
    assert verify_report(report, store, acc)["passed"]


def test_checks_must_match_tools_actually_run(store):
    acc, rows = _txns(store, 2)
    claims = [{"statement": "Payment.", "txn_ids": [t], "accounts": [acc]} for t in rows.txn_id]
    report = {"verdict": "suspicious", "claims": claims,
              "checks": [{"description": "No loops found.", "tool": "find_cycles"}]}
    assert verify_report(report, store, acc, tools_used={"find_cycles"})["passed"]
    failed = verify_report(report, store, acc, tools_used={"account_profile"})
    assert not failed["passed"] and failed["checks_verified"] == 0


def test_exact_figures_must_be_exact_but_rounded_ones_may_round(store):
    acc, rows = _txns(store, 2)
    total = float(rows.amount.sum())
    def report(text):
        return {"verdict": "not_suspicious", "summary": "", "claims": [
            {"statement": text, "txn_ids": list(rows.txn_id), "accounts": [acc]}]}
    from parakh.fmt import inr
    assert verify_report(report(f"Sent {inr(total)} in total."), store, acc)["claims_verified"] == 1
    assert verify_report(report(f"Sent {inr(total + 5934)} in total."), store, acc)["claims_verified"] == 0
    if total >= 1e5:
        assert verify_report(report(f"Sent about ₹{total / 1e5:.1f} lakh."), store, acc)["claims_verified"] == 1


def test_summary_amounts_must_be_supported(store):
    acc, rows = _txns(store, 2)
    claims = [{"statement": "Payment.", "txn_ids": [t], "accounts": [acc]} for t in rows.txn_id]
    bad = {"verdict": "suspicious", "summary": "It moved ₹12,34,56,789 overall.", "claims": claims}
    assert any(i.startswith("summary") for i in verify_report(bad, store, acc)["issues"])


def test_counts_in_words_are_checked_but_amount_digits_are_not(store):
    from parakh.agents.verifier import count_mentions
    acc, rows = _txns(store, 2)
    assert count_mentions(f"{acc} sent ₹1,234.00 in two payments.", rows, acc) == []
    wrong = count_mentions(f"{acc} made three payments.", rows, acc)
    assert wrong and wrong[0]["expected"] == 2 and wrong[0]["replacement"] == "two"
