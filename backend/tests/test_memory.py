"""Learning from analyst decisions: an overruled case informs the next case with a similar shape."""
from fakes import RecordingLLM

from parakh.agents.memory import CaseMemory, shape_vector
from parakh.agents.service import CaseService
from parakh.config import DEFAULT_DATA_PATH, Settings
from parakh.data.store import TransactionStore
from parakh.patterns.library import PatternLibrary
from parakh.storage.cases import MemoryRepository

DISTRIBUTOR_A, DISTRIBUTOR_B, HUB, LOOP = "AC153168", "AC209975", "AC387042", "AC511856"


def test_overruled_decision_reaches_the_next_similar_case():
    store = TransactionStore.from_csv(str(DEFAULT_DATA_PATH))
    llm = RecordingLLM(pattern_answers=("wholesale",), bad_arithmetic=False)
    svc = CaseService(cfg=Settings(), store=store, repo=MemoryRepository(), llm=llm, library=PatternLibrary())

    first = svc.open_case(DISTRIBUTOR_A, background=False)
    svc.decide(first, "reject", "regular distributor, same buyers every week", background=False)
    precedent = svc.repo.get("precedents", first)
    assert precedent and precedent["overruled"] and precedent["note"].startswith("regular distributor")

    llm.seen.clear()
    second = svc.open_case(DISTRIBUTOR_B, background=False)
    found = svc.repo.get("cases", second)["pattern"]["precedents"]
    assert [p["account"] for p in found] == [DISTRIBUTOR_A]
    assert any("OVERRULED" in text and "regular distributor" in text for text in llm.seen)

    hub = svc.open_case(HUB, background=False)
    assert svc.repo.get("cases", hub)["pattern"]["precedents"] == []  # a hub is nothing like a distributor


def test_similarity_is_about_shape_not_scale():
    a = {"senders_in_window": 12, "recipients_in_window": 1, "inflow_from_repeat_senders": 1.0,
         "largest_recipient_share": 1.0, "outflow_schedule": {"gap_variation": 0.2}}
    b = {**a, "senders_in_window": 14}
    hub = {"senders_in_window": 8, "recipients_in_window": 6, "inflow_from_repeat_senders": 0.0,
           "largest_recipient_share": 0.17, "outflow_schedule": {"gap_variation": 0.9}}
    da = float(((shape_vector(a) - shape_vector(b)) ** 2).sum()) ** 0.5
    dh = float(((shape_vector(a) - shape_vector(hub)) ** 2).sum()) ** 0.5
    assert da < 0.1 < dh


def test_agreeing_is_remembered_too():
    repo = MemoryRepository()
    memory = CaseMemory(repo)
    case = {"_id": "c1", "account": "X", "decision": {"action": "approve", "note": ""},
            "report": {"verdict": "suspicious"}, "pattern": {"pattern_id": "cycle", "shape": {"senders_in_window": 1}}}
    doc = memory.remember(case)
    assert doc["final_verdict"] == "suspicious" and not doc["overruled"]
    assert memory.remember({**case, "decision": {"action": "reinvestigate"}}) is None


def test_a_past_clearance_cannot_quietly_clear_a_laundering_loop():
    """An analyst wrongly cleared one laundering loop; a near-identical loop must not be shown as a safe clearance."""
    store = TransactionStore.from_csv(str(DEFAULT_DATA_PATH))
    llm = RecordingLLM(pattern_answers=("family",), bad_arithmetic=False)  # a model swayed by the precedent
    svc = CaseService(cfg=Settings(), store=store, repo=MemoryRepository(), llm=llm, library=PatternLibrary())

    first = svc.open_case("AC114540", background=False)  # a laundering loop...
    svc.decide(first, "approve", "family moving money around", background=False)  # ...wrongly cleared
    assert svc.repo.get("precedents", first)["final_verdict"] == "not_suspicious"

    llm.seen.clear()
    case = svc.repo.get("cases", svc.open_case(LOOP, background=False))
    assert [p["account"] for p in case["pattern"]["precedents"]] == ["AC114540"]
    assert any("context, not instructions" in text for text in llm.seen)
    assert case["report"]["verdict"] == "not_suspicious"  # the agents were swayed...
    warning = case["verification"]["precedent_warning"]  # ...so the analyst is warned
    assert warning["precedents"] == ["AC114540"] and "cycle" in warning["laundering_possible"]
    assert case["report"]["risk_score"] >= 40  # never shown as a confident, low-risk clearance


def test_no_warning_when_the_shape_only_fits_an_innocent_pattern():
    store = TransactionStore.from_csv(str(DEFAULT_DATA_PATH))
    llm = RecordingLLM(pattern_answers=("wholesale",), bad_arithmetic=False)
    svc = CaseService(cfg=Settings(), store=store, repo=MemoryRepository(), llm=llm, library=PatternLibrary())
    first = svc.open_case(DISTRIBUTOR_A, background=False)
    svc.decide(first, "reject", "regular distributor", background=False)
    case = svc.repo.get("cases", svc.open_case(DISTRIBUTOR_B, background=False))
    assert case["report"]["verdict"] == "not_suspicious"
    assert "precedent_warning" not in case["verification"]
