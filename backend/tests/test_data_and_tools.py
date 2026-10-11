import re

from conftest import first_txn

from parakh.agents.service import load_or_screen
from parakh.data.generator import PATTERNS
from parakh.fmt import inr
from parakh.tools.investigation import InvestigationTools
from parakh.tools.screening import screen


def test_generator_labels_every_pattern(small_df):
    found = set(small_df[small_df.is_laundering == 1].pattern)
    assert found == set(PATTERNS)
    assert small_df.txn_id.is_unique


def test_account_ids_do_not_leak_roles(small_df):
    ids = set(small_df.from_account) | set(small_df.to_account)
    assert all(re.fullmatch(r"AC\d{6}", a) for a in ids)


def test_agents_cannot_see_labels(store):
    assert "is_laundering" not in store.df.columns
    assert "pattern" not in store.df.columns


def test_screening_flags_laundering_and_innocent_accounts(store):
    alerts = screen(store)
    flagged = {a.account for a in alerts}
    bad = set(store.laundering_accounts)
    assert flagged & bad, "screening should catch some laundering"
    assert flagged - bad, "screening should also raise false alarms (that is what triage is for)"


def test_alert_cache_round_trip(store, data_file):
    first = load_or_screen(store, data_file)
    second = load_or_screen(store, data_file)
    assert first == second and len(first) > 0


def test_account_profile(store):
    tools = InvestigationTools(store)
    acc = first_txn(store, "fan_out").from_account
    prof = tools.account_profile(acc)
    assert prof["account"] == acc and prof["outgoing"]["count"] > 0
    assert "error" in tools.account_profile("AC000000")


def test_find_cycles_detects_round_trip(store):
    tools = InvestigationTools(store)
    acc = first_txn(store, "cycle").to_account
    cycles = tools.find_cycles(acc)["cycles"]
    assert any(c["within_window"] and len(c["accounts"]) >= 3 for c in cycles)


def test_common_counterparties_detects_scatter_gather(store):
    tools = InvestigationTools(store)
    src = first_txn(store, "scatter_gather").to_account  # first hop: external source -> scatter account
    shared = tools.common_counterparties(src, "out")["shared_accounts"]
    assert any(v["reached_via"] >= 3 for v in shared.values())


def test_trace_money_follows_layering_chain(store):
    tools = InvestigationTools(store)
    start = first_txn(store, "layering_chain").to_account
    trace = tools.trace_money(start, "out", hops=3)
    assert trace["hops_reached"] >= 2
    assert all("txn_id" in t for t in trace["transfers"])


def test_list_transactions_is_capped(store):
    tools = InvestigationTools(store)
    acc = store.df.from_account.value_counts().index[0]
    res = tools.list_transactions(acc, limit=500)
    assert res["shown"] <= 20


def test_indian_rupee_format():
    assert inr(6641114) == "₹66,41,114"
    assert inr(999) == "₹999"
    assert inr(123456789) == "₹12,34,56,789"
