import time

import mongomock
import pandas as pd
from fastapi.testclient import TestClient
from fakes import FakeLLM

from parakh.agents.service import CaseService
from parakh.api.main import create_app
from parakh.config import Settings
from parakh.data.ibm_loader import convert
from parakh.patterns.library import PatternLibrary
from parakh.storage.cases import MemoryRepository, MongoRepository


def client(store):
    svc = CaseService(cfg=Settings(), store=store, repo=MemoryRepository(), llm=FakeLLM(), library=PatternLibrary())
    return TestClient(create_app(svc, require_llm_key=False))


def wait(c, url, done):
    for _ in range(100):
        body = c.get(url).json()
        if done(body):
            return body
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting on {url}")


def test_health_and_stats(store):
    with client(store) as c:
        assert c.get("/api/health").json()["status"] == "ok"
        stats = c.get("/api/stats").json()
        assert stats["transactions"] == len(store.df) and stats["alerts"] > 0


def test_run_review_and_decide(store):
    with client(store) as c:
        run = c.post("/api/runs", json={"max_cases": 2}).json()
        wait(c, f"/api/runs/{run['_id']}", lambda r: r["status"] in ("done", "failed"))
        cases = c.get("/api/cases").json()
        assert len(cases) == 2
        cid = cases[0]["_id"]
        assert c.get(f"/api/cases/{cid}").json()["report"]["claims"]
        assert len(c.get(f"/api/cases/{cid}/transactions").json()) > 0
        assert c.post(f"/api/cases/{cid}/decision", json={"action": "reinvestigate"}).status_code == 422
        assert c.post(f"/api/cases/{cid}/decision", json={"action": "approve"}).status_code == 200
        case = wait(c, f"/api/cases/{cid}", lambda x: x["status"] not in ("closing", "investigating"))
        assert case["status"] == "confirmed_suspicious"
        assert c.post(f"/api/cases/{cid}/decision", json={"action": "approve"}).status_code == 409


def test_open_case_for_unknown_account(store):
    with client(store) as c:
        assert c.post("/api/cases", json={"account": "AC000000"}).status_code == 404
        assert c.get("/api/cases/nope").status_code == 404


def test_mongo_repository():
    repo = MongoRepository("mongodb://unused", "parakh_test", client=mongomock.MongoClient())
    repo.insert("cases", {"_id": "c1", "created_at": "1", "status": "queued"})
    repo.update("cases", "c1", {"status": "awaiting_review"})
    assert repo.get("cases", "c1")["status"] == "awaiting_review"
    assert [d["_id"] for d in repo.list("cases")] == ["c1"]


def test_ibm_loader(tmp_path):
    trans = tmp_path / "t.csv"
    pd.DataFrame({
        "Timestamp": ["2022/09/01 00:20", "2022/09/01 00:25", "2022/09/01 01:00"],
        "From Bank": [10, 10, 20], "Account": ["A1", "A1", "B2"],
        "To Bank": [20, 30, 10], "Account.1": ["B2", "C3", "A1"],
        "Amount Received": [100.0, 200.0, 50.0], "Receiving Currency": ["US Dollar"] * 3,
        "Amount Paid": [100.0, 200.0, 50.0], "Payment Currency": ["US Dollar"] * 3,
        "Payment Format": ["ACH", "ACH", "Cheque"], "Is Laundering": [1, 1, 0],
    }).to_csv(trans, index=False)
    patterns = tmp_path / "p.txt"
    patterns.write_text(
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out\n"
        "2022/09/01 00:20,10,A1,20,B2,100.00,US Dollar,100.00,US Dollar,ACH,1\n"
        "END LAUNDERING ATTEMPT - FAN-OUT\n")
    df = convert(str(trans), str(patterns), days=10, max_rows=None)
    assert list(df.columns[:3]) == ["txn_id", "timestamp", "from_account"]
    assert df.loc[df.from_account == "10-A1"].iloc[0].pattern == "fan_out"
    assert (df.pattern == "unlabelled").sum() == 1


def test_case_flow(store):
    with client(store) as c:
        run = c.post("/api/runs", json={"max_cases": 1}).json()
        wait(c, f"/api/runs/{run['_id']}", lambda r: r["status"] in ("done", "failed"))
        cid = c.get("/api/cases").json()[0]["_id"]
        flow = c.get(f"/api/cases/{cid}/flow").json()
        assert flow["inflows"] or flow["outflows"]
        assert flow["total_in"] >= 0 and 0 <= flow["passed_through"] <= 1
        assert any(p["cited"] for p in flow["inflows"] + flow["outflows"])


def test_evaluations_endpoint(store, tmp_path, monkeypatch):
    import json as _json

    import parakh.api.main as api_main
    (tmp_path / "x.json").write_text(_json.dumps({"name": "x", "updated_at": "2026-01-01", "results": {}}))
    monkeypatch.setattr(api_main, "PUBLISHED_DIR", tmp_path)
    with client(store) as c:
        assert [e["name"] for e in c.get("/api/evaluations").json()] == ["x"]
