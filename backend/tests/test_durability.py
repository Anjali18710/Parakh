"""Cases and paused agent runs survive a server restart when stored in MongoDB."""
from fakes import RecordingLLM
from mongo_shim import MongoClient

from parakh.agents.service import CaseService
from parakh.config import DEFAULT_DATA_PATH, Settings
from parakh.data.store import TransactionStore
from parakh.patterns.library import PatternLibrary
from parakh.storage.cases import MongoRepository

HUB = "AC387042"


def service(client, store):
    repo = MongoRepository("mongodb://test", "parakh_test", client=client)
    return CaseService(cfg=Settings(), store=store, repo=repo,
                       llm=RecordingLLM(("gather_scatter",), bad_arithmetic=False), library=PatternLibrary())


def test_paused_case_and_interrupted_case_after_restart():
    client, store = MongoClient(), TransactionStore.from_csv(str(DEFAULT_DATA_PATH))
    before = service(client, store)
    paused = before.open_case(HUB, background=False)
    assert before.repo.get("cases", paused)["status"] == "awaiting_review"
    # The server dies while the analyst's decision is being applied, and while another case is running.
    before.repo.update("cases", paused, {"status": "closing"})
    stuck = before.create_case(HUB)
    before.repo.update("cases", stuck, {"status": "investigating"})

    after = service(client, store)  # a fresh process on the same database
    assert after.recover() == {"restored": 1, "interrupted": 1}
    assert after.repo.get("cases", paused)["status"] == "awaiting_review"
    assert "restart" in after.repo.get("cases", stuck)["error"]

    # The paused agent run itself survived: the decision resumes it rather than being applied by hand.
    after.decide(paused, "approve", background=False)
    case = after.repo.get("cases", paused)
    assert case["status"] == "confirmed_suspicious"
    assert any(e["agent"] == "reviewer" for e in case["timeline"])

    # The interrupted case can be retried from scratch.
    after.retry(stuck, background=False)
    assert after.repo.get("cases", stuck)["status"] == "awaiting_review"
