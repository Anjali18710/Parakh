"""Structured outputs the agents must produce."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TriageDecision(BaseModel):
    account: str
    open_case: bool = Field(description="True if this alert deserves a full investigation.")
    priority: int = Field(ge=1, le=5, description="1 = most urgent, 5 = least urgent.")
    reason: str = Field(description="One sentence explaining the decision.")


class TriageResult(BaseModel):
    decisions: list[TriageDecision]


class PatternFinding(BaseModel):
    pattern_id: str = Field(description="Id of the best matching pattern from the candidates, or 'none'.")
    kind: Literal["laundering", "innocent", "unclear"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(description="Two or three sentences comparing the evidence with the pattern's tell-tale checks.")
    ruled_out: list[str] = Field(default_factory=list, description="Ids of candidate patterns that were ruled out.")


class Claim(BaseModel):
    statement: str = Field(description="One factual sentence about the money flow.")
    txn_ids: list[str] = Field(description="Transaction ids that prove this statement.")
    accounts: list[str] = Field(default_factory=list, description="Accounts mentioned in the statement.")
    amount: float | None = Field(default=None, description="Total amount in rupees stated in the claim, if any.")
    count: int | None = Field(default=None, description="Number of transactions or accounts stated, if any.")


class CaseReport(BaseModel):
    verdict: Literal["suspicious", "not_suspicious"]
    risk_score: int = Field(ge=0, le=100)
    pattern_id: str = Field(description="Pattern id from the pattern finding, or 'none'.")
    summary: str = Field(description="Three to five sentences an analyst can read in under a minute.")
    claims: list[Claim] = Field(description="Every factual claim the verdict relies on, each with evidence.")
    recommended_action: str = Field(description="What the analyst should do next.")
