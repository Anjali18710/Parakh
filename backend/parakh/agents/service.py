"""Runs investigations end to end: screening -> triage -> per-case agent graph -> human review."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from functools import cached_property
from pathlib import Path

from langgraph.types import Command

from parakh.agents.graph import Deps, build_case_graph
from parakh.agents.memory import CaseMemory
from parakh.agents.llm import UsageTracker, get_llm
from parakh.agents.triage import brief, triage_by_score, triage_with_llm
from parakh.config import Settings, settings as default_settings
from parakh.data.store import TransactionStore
from parakh.patterns.library import PatternLibrary
from parakh.storage.cases import CaseRepository, make_checkpointer, make_repository, now
from parakh.tools.investigation import InvestigationTools
from parakh.tools.screening import screen

log = logging.getLogger("parakh")


def alerts_cache_path(data_path: str) -> Path:
    return Path(str(data_path).replace(".csv.gz", "").replace(".csv", "") + ".alerts.json")


def load_or_screen(store: TransactionStore, data_path: str) -> list[dict]:
    """Screening the whole dataset takes a while on small servers, so results are cached next to the data."""
    cache = alerts_cache_path(data_path)
    fingerprint = Path(data_path).stat().st_size  # file times change on git clone; size does not
    try:
        if cache.exists():
            cached = json.loads(cache.read_text())
            if cached.get("source_size") == fingerprint:
                return cached["alerts"]
    except (OSError, ValueError, AttributeError):
        pass
    alerts = [a.to_dict() for a in screen(store)]
    try:
        cache.write_text(json.dumps({"source_size": fingerprint, "alerts": alerts}))
    except OSError:
        log.warning("could not write alert cache to %s", cache)
    return alerts

OPEN_STATUSES = {"queued", "investigating", "awaiting_review"}
FINAL_STATUSES = {"confirmed_suspicious", "closed_not_suspicious"}


class CaseService:
    def __init__(self, cfg: Settings | None = None, store: TransactionStore | None = None,
                 repo: CaseRepository | None = None, llm=None, library: PatternLibrary | None = None):
        self.cfg = cfg or default_settings
        self._store = store
        self._store_from_file = store is None
        self.repo = repo or make_repository(self.cfg.mongodb_uri, self.cfg.mongodb_db)
        self._llm = llm
        self._library = library
        self.checkpointer = make_checkpointer(self.repo)
        self._graphs: dict[str, object] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ lazy resources
    @property
    def store(self) -> TransactionStore:
        if self._store is None:
            self._store = TransactionStore.from_csv(self.cfg.data_path)
        return self._store

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(self.cfg)
        return self._llm

    @property
    def library(self) -> PatternLibrary:
        if self._library is None:
            self._library = PatternLibrary()
        return self._library

    @cached_property
    def memory(self) -> CaseMemory:
        return CaseMemory(self.repo)

    @cached_property
    def itools(self) -> InvestigationTools:
        return InvestigationTools(self.store)

    @cached_property
    def alerts(self) -> list[dict]:
        if self._store_from_file:
            return load_or_screen(self.store, self.cfg.data_path)
        return [a.to_dict() for a in screen(self.store)]  # injected store: the cache file is not for it

    def alert_for(self, account: str) -> dict:
        for a in self.alerts:
            if a["account"] == account:
                return a
        return {"account": account, "score": 0, "signals": []}

    def graph(self, mode: str = "multi"):
        with self._lock:
            if mode not in self._graphs:
                deps = Deps(self.llm, self.store, self.itools, self.library, self.cfg, self.memory)
                self._graphs[mode] = build_case_graph(deps, mode=mode, human_review=True,
                                                      enforce_verifier=self.cfg.verifier_enabled,
                                                      checkpointer=self.checkpointer)
            return self._graphs[mode]

    # ------------------------------------------------------------ runs
    def start_run(self, max_cases: int, mode: str = "multi", use_llm_triage: bool = True,
                  background: bool = True) -> dict:
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        run = {"_id": run_id, "created_at": now(), "status": "triaging", "mode": mode,
               "max_cases": max_cases, "alerts_screened": len(self.alerts), "case_ids": []}
        self.repo.insert("runs", run)
        job = lambda: self._run(run_id, max_cases, mode, use_llm_triage)  # noqa: E731
        if background:
            threading.Thread(target=job, daemon=True).start()
        else:
            job()
        return self.repo.get("runs", run_id)

    def _run(self, run_id: str, max_cases: int, mode: str, use_llm_triage: bool) -> None:
        try:
            busy = {c["account"] for c in self.repo.list("cases", 1000) if c.get("status") in OPEN_STATUSES}
            candidates = [a for a in self.alerts if a["account"] not in busy]
            # The triage agent sees the strongest alerts first; the rest wait for the next run.
            pool = candidates[: max(60, max_cases * 4)]
            briefs = [brief(a) for a in pool]
            tracker = UsageTracker()
            if use_llm_triage:
                decisions = triage_with_llm(self.llm, briefs, max_cases, {"callbacks": [tracker]})
            else:
                decisions = triage_by_score(briefs, max_cases)
            selected = [d for d in decisions if d["selected"]]
            self.repo.update("runs", run_id, {"status": "investigating", "triage": decisions,
                                              "triage_usage": tracker.summary(self.cfg)})
            case_ids = []
            for d in selected:
                case_id = self.create_case(d["account"], mode, triage=d, run_id=run_id)
                case_ids.append(case_id)
            self.repo.update("runs", run_id, {"case_ids": case_ids})
            for case_id in case_ids:
                self.investigate(case_id)
            self.repo.update("runs", run_id, {"status": "done", "finished_at": now()})
        except Exception as exc:  # surface failures in the UI instead of dying silently
            log.exception("run %s failed", run_id)
            self.repo.update("runs", run_id, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    # ------------------------------------------------------------ cases
    def create_case(self, account: str, mode: str = "multi", triage: dict | None = None,
                    run_id: str | None = None, notes: list[str] | None = None) -> str:
        case_id = f"case_{uuid.uuid4().hex[:10]}"
        self.repo.insert("cases", {
            "_id": case_id, "case_id": case_id, "account": account, "mode": mode, "run_id": run_id,
            "status": "queued", "created_at": now(), "updated_at": now(), "alert": self.alert_for(account),
            "triage": triage, "analyst_notes": notes or [], "timeline": [], "usage": None,
        })
        return case_id

    def open_case(self, account: str, mode: str = "multi", note: str | None = None, background: bool = True) -> str:
        if not self.store.has_account(account):
            raise KeyError(account)
        case_id = self.create_case(account, mode, notes=[note] if note else None)
        if background:
            threading.Thread(target=self.investigate, args=(case_id,), daemon=True).start()
        else:
            self.investigate(case_id)
        return case_id

    def investigate(self, case_id: str, resume: dict | None = None) -> None:
        case = self.repo.get("cases", case_id)
        tracker = UsageTracker()
        live = list(case.get("timeline") or [])

        def event_sink(event: dict) -> None:
            # Save each agent step as it happens, so the dashboard can show the trail filling in live.
            live.append(event)
            self.repo.update("cases", case_id, {"timeline": live, "updated_at": now()})

        thread = case.get("thread_id") or case_id
        config = {"configurable": {"thread_id": thread, "event_sink": event_sink},
                  "callbacks": [tracker], "recursion_limit": 60}
        graph = self.graph(case.get("mode", "multi"))
        if resume is None or resume.get("action") == "reinvestigate":
            self.repo.update("cases", case_id, {"status": "investigating", "updated_at": now()})
        started = time.time()
        try:
            if resume is not None:
                graph.invoke(Command(resume=resume), config)
            else:
                graph.invoke({"case_id": case_id, "account": case["account"], "alert": case["alert"],
                              "mode": case.get("mode", "multi"), "analyst_notes": case.get("analyst_notes", []),
                              "evidence": [], "timeline": [], "tool_calls": 0}, config)
            snap = graph.get_state(config)
            values = snap.values
            status = "awaiting_review" if snap.next else values.get("status", "closed")
            prev = case.get("usage") or {}
            usage = tracker.summary(self.cfg)
            for k in ("llm_calls", "input_tokens", "output_tokens", "est_cost_usd"):
                usage[k] = round(usage[k] + prev.get(k, 0), 5)
            self.repo.update("cases", case_id, {
                "status": status, "updated_at": now(),
                "report": values.get("report"), "pattern": values.get("pattern"),
                "verification": values.get("verification"), "timeline": values.get("timeline", []),
                "investigation_summary": values.get("investigation_summary"),
                "injections": values.get("injections", []),
                "tool_calls": values.get("tool_calls", 0), "decision": values.get("decision"),
                "analyst_notes": values.get("analyst_notes", []),
                "usage": usage, "duration_s": round((case.get("duration_s") or 0) + time.time() - started, 1),
            })
            if status in FINAL_STATUSES:
                self.memory.remember(self.repo.get("cases", case_id))
        except Exception as exc:
            log.exception("case %s failed", case_id)
            self.repo.update("cases", case_id, {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                                "updated_at": now()})

    # ------------------------------------------------------------ restarts
    def recover(self) -> dict:
        """After a restart: restore cases that were waiting for review, mark half-finished ones as interrupted."""
        restored = interrupted = 0
        for case in self.repo.list("cases", 5000):
            if case.get("status") not in ("queued", "investigating", "closing"):
                continue
            snap = self.graph(case.get("mode", "multi")).get_state(
                {"configurable": {"thread_id": case.get("thread_id") or case["_id"]}})
            if snap.next and "human_review" in snap.next:
                self.repo.update("cases", case["_id"], {"status": "awaiting_review", "updated_at": now()})
                restored += 1
            else:
                self.repo.update("cases", case["_id"], {
                    "status": "failed", "updated_at": now(),
                    "error": "Interrupted by a server restart before the agents finished. Retry to run it again."})
                interrupted += 1
        for run in self.repo.list("runs", 500):
            if run.get("status") in ("triaging", "investigating"):
                self.repo.update("runs", run["_id"], {"status": "failed", "error": "Interrupted by a server restart"})
        return {"restored": restored, "interrupted": interrupted}

    def retry(self, case_id: str, background: bool = True) -> dict:
        """Run a failed or interrupted case again from the start, on a fresh agent thread."""
        case = self.repo.get("cases", case_id)
        if case is None:
            raise KeyError(case_id)
        if case.get("status") != "failed":
            raise ValueError(f"case is {case['status']}; only failed cases can be retried")
        attempt = int(case.get("retries", 0)) + 1
        self.repo.update("cases", case_id, {
            "status": "queued", "error": None, "retries": attempt, "thread_id": f"{case_id}-retry{attempt}",
            "timeline": [], "report": None, "pattern": None, "verification": None, "usage": None,
            "duration_s": 0, "updated_at": now()})
        if background:
            threading.Thread(target=self.investigate, args=(case_id,), daemon=True).start()
        else:
            self.investigate(case_id)
        return self.repo.get("cases", case_id)

    def decide(self, case_id: str, action: str, note: str | None = None, background: bool = True) -> dict:
        case = self.repo.get("cases", case_id)
        if case is None:
            raise KeyError(case_id)
        if case["status"] != "awaiting_review":
            raise ValueError(f"case is {case['status']}, not awaiting review")
        decision = {"action": action, "note": note, "at": now()}
        graph = self.graph(case.get("mode", "multi"))
        snap = graph.get_state({"configurable": {"thread_id": case.get("thread_id") or case_id}})
        if snap.next:  # normal path: resume the paused LangGraph run
            if background:
                self.repo.update("cases", case_id, {"status": "investigating" if action == "reinvestigate" else "closing"})
                threading.Thread(target=self.investigate, args=(case_id, decision), daemon=True).start()
            else:
                self.investigate(case_id, decision)
        else:  # server restarted and the paused run is gone: apply the decision directly
            verdict = (case.get("report") or {}).get("verdict")
            if action == "reinvestigate":
                new_id = self.open_case(case["account"], case.get("mode", "multi"), note, background)
                self.repo.update("cases", case_id, {"status": "superseded", "superseded_by": new_id})
            else:
                agree = action == "approve"
                status = "confirmed_suspicious" if (verdict == "suspicious") == agree else "closed_not_suspicious"
                self.repo.update("cases", case_id, {"status": status, "decision": decision, "updated_at": now()})
                self.memory.remember(self.repo.get("cases", case_id))
        return self.repo.get("cases", case_id)
