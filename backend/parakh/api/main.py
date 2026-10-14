"""Parakh HTTP API.

Run locally:  uvicorn parakh.api.main:app --reload
"""
from __future__ import annotations

import logging
import os
import threading
from collections import Counter
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from parakh.agents.service import CaseService
from parakh.config import settings
from parakh.data.remarks import injection_reasons
from parakh.reports.str_pdf import build_str, filename as str_filename
from parakh.tools.flow import case_flow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class RunRequest(BaseModel):
    max_cases: int = Field(default=5, ge=1, le=50)
    mode: Literal["multi", "single"] = "multi"
    use_llm_triage: bool = True


class OpenCaseRequest(BaseModel):
    account: str
    mode: Literal["multi", "single"] = "multi"
    note: str | None = None


class DecisionRequest(BaseModel):
    action: Literal["approve", "reject", "reinvestigate"]
    note: str | None = Field(default=None, max_length=1000)


def llm_configured() -> bool:
    key = "GOOGLE_API_KEY" if settings.llm_provider == "gemini" else "GROQ_API_KEY"
    return bool(os.getenv(key))


def create_app(service: CaseService | None = None, require_llm_key: bool = True) -> FastAPI:
    svc = service or CaseService()
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Loading data and running screening takes a few seconds; do it before the first request needs it.
        threading.Thread(target=lambda: svc.alerts, daemon=True).start()
        try:
            logging.getLogger("parakh").info("recovered after restart: %s", svc.recover())
        except Exception:  # never block startup on recovery
            logging.getLogger("parakh").exception("recovery after restart failed")
        yield

    app = FastAPI(title="Parakh", description="Agentic anti-money-laundering investigations", version="1.0.0",
                  lifespan=lifespan)
    app.state.service = svc
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

    def need_llm():
        if require_llm_key and not llm_configured():
            raise HTTPException(503, "No LLM API key configured on the server (set GROQ_API_KEY).")

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "llm_provider": settings.llm_provider,
            "model": settings.gemini_model if settings.llm_provider == "gemini" else settings.groq_model,
            "llm_configured": llm_configured() or not require_llm_key,
            "storage": "mongodb" if settings.mongodb_uri else "memory",
            "verifier_enabled": settings.verifier_enabled,
        }

    @app.get("/api/stats")
    def stats():
        cases = svc.repo.list("cases", 1000)
        usage = [c["usage"] for c in cases if c.get("usage")]
        durations = [c["duration_s"] for c in cases if c.get("duration_s")]
        verif = [c["verification"] for c in cases if c.get("verification")]
        return {
            "transactions": len(svc.store.df),
            "accounts": len(set(svc.store.df.from_account) | set(svc.store.df.to_account)),
            "alerts": len(svc.alerts),
            "cases": len(cases),
            "by_status": dict(Counter(c["status"] for c in cases)),
            "avg_cost_usd": round(sum(u["est_cost_usd"] for u in usage) / len(usage), 5) if usage else None,
            "avg_duration_s": round(sum(durations) / len(durations), 1) if durations else None,
            "claims_verified": sum(v["claims_verified"] for v in verif),
            "claims_total": sum(v["claims_total"] for v in verif),
        }

    @app.get("/api/alerts")
    def alerts(limit: int = 50):
        open_accounts = {c["account"]: c["_id"] for c in svc.repo.list("cases", 1000)}
        return [{**{k: a[k] for k in ("account", "score")},
                 "signals": [{"name": s["name"], "detail": s["detail"]} for s in a["signals"]],
                 "case_id": open_accounts.get(a["account"])} for a in svc.alerts[:limit]]

    @app.post("/api/runs")
    def start_run(req: RunRequest):
        need_llm()
        return svc.start_run(min(req.max_cases, settings.max_cases_per_run), req.mode, req.use_llm_triage)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        run = svc.repo.get("runs", run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        return run

    @app.get("/api/cases")
    def list_cases():
        keep = ("_id", "account", "status", "mode", "created_at", "updated_at", "alert", "duration_s", "error")
        out = []
        for c in svc.repo.list("cases", 500):
            item = {k: c.get(k) for k in keep}
            r = c.get("report") or {}
            v = c.get("verification") or {}
            item.update({"verdict": r.get("verdict"), "risk_score": r.get("risk_score"),
                         "pattern_id": r.get("pattern_id"), "claims_verified": v.get("claims_verified"),
                         "claims_total": v.get("claims_total")})
            item["alert_score"] = (c.get("alert") or {}).get("score")
            item.pop("alert")
            out.append(item)
        return out

    @app.post("/api/cases")
    def open_case(req: OpenCaseRequest):
        need_llm()
        try:
            case_id = svc.open_case(req.account.strip(), req.mode, req.note)
        except KeyError:
            raise HTTPException(404, f"Account {req.account} not found")
        return svc.repo.get("cases", case_id)

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str):
        case = svc.repo.get("cases", case_id)
        if not case:
            raise HTTPException(404, "Case not found")
        return case

    @app.get("/api/cases/{case_id}/transactions")
    def case_transactions(case_id: str):
        case = svc.repo.get("cases", case_id)
        if not case:
            raise HTTPException(404, "Case not found")
        ids: list[str] = []
        for claim in (case.get("report") or {}).get("claims", []):
            ids += claim.get("txn_ids", [])
        rows = svc.store.get_txns(list(dict.fromkeys(ids)))
        return [{"txn_id": r.txn_id, "timestamp": str(r.timestamp), "from_account": r.from_account,
                 "to_account": r.to_account, "amount": float(r.amount), "payment_format": r.payment_format,
                 "remarks": r.remarks, "remark_flagged": bool(injection_reasons(r.remarks))}
                for r in rows.sort_values("timestamp").itertuples()]

    @app.get("/api/cases/{case_id}/flow")
    def case_money_flow(case_id: str):
        case = svc.repo.get("cases", case_id)
        if not case:
            raise HTTPException(404, "Case not found")
        ids = [t for c in (case.get("report") or {}).get("claims", []) for t in c.get("txn_ids", [])]
        return case_flow(svc.store, case["account"], ids)

    @app.get("/api/cases/{case_id}/str.pdf")
    def suspicious_transaction_report(case_id: str):
        """FIU-IND style STR for the case: a watermarked draft until an analyst confirms it."""
        case = svc.repo.get("cases", case_id)
        if not case:
            raise HTTPException(404, "Case not found")
        verdict = (case.get("report") or {}).get("verdict")
        if case["status"] != "confirmed_suspicious" and not (case["status"] == "awaiting_review" and verdict == "suspicious"):
            raise HTTPException(409, "An STR is only prepared for a case the agents or an analyst found suspicious.")
        pdf = build_str(case, svc.store, settings.reporting_entity)
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{str_filename(case)}"'})

    @app.post("/api/cases/{case_id}/retry")
    def retry(case_id: str):
        need_llm()
        try:
            return svc.retry(case_id)
        except KeyError:
            raise HTTPException(404, "Case not found")
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/cases/{case_id}/decision")
    def decide(case_id: str, req: DecisionRequest):
        if req.action == "reinvestigate":
            need_llm()
            if not (req.note or "").strip():
                raise HTTPException(422, "Tell the agents what to look into.")
        try:
            return svc.decide(case_id, req.action, req.note)
        except KeyError:
            raise HTTPException(404, "Case not found")
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    return app


app = create_app()
