"""Library of laundering typologies (and innocent look-alikes) with FAISS search.

The pattern agent describes what it found in plain words; FAISS returns the
closest known patterns, including the *innocent* ones that commonly cause
false alarms, so the agent can rule them out explicitly.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass(frozen=True)
class Typology:
    id: str
    name: str
    kind: str  # "laundering" | "innocent"
    description: str
    tell_tale_checks: str


TYPOLOGIES: list[Typology] = [
    Typology("fan_out", "Fan-out (smurfing out)", "laundering",
             "One account receives a large sum and quickly splits it into many similar-sized transfers to "
             "many different accounts within hours, often just under reporting thresholds.",
             "Large single inflow followed within a day by many outflows of similar size to new recipients; "
             "recipients have little history with the sender; almost all of the inflow leaves."),
    Typology("fan_in", "Fan-in (collection)", "laundering",
             "Many accounts each send a sizeable amount to one collector account in a short time; the collector "
             "then moves the pooled money out in one or two large transfers.",
             "Many distinct senders within hours or a day, amounts similar and large, senders were themselves "
             "funded just before, collector forwards nearly everything soon after."),
    Typology("cycle", "Round-tripping cycle", "laundering",
             "Money moves through a loop of accounts and returns to the starting account within days, losing a "
             "small fee at each hop, to create fake turnover or disguise the origin.",
             "A loop of three or more accounts closed within a few days; each hop slightly smaller than the last; "
             "the accounts have little other activity."),
    Typology("scatter_gather", "Scatter-gather", "laundering",
             "A source splits money across several intermediary accounts, and each intermediary forwards almost "
             "the same amount to one common destination.",
             "The recipients of one account all pay the same destination soon after; amounts at each step almost "
             "match; intermediaries are pass-through only."),
    Typology("gather_scatter", "Gather-scatter (hub)", "laundering",
             "Many accounts pay into a hub account, which then quickly pays out to many other accounts, "
             "breaking the link between where money came from and where it goes.",
             "A hub with many distinct senders and many distinct recipients in a short period, inflow and outflow "
             "totals nearly equal, little or no balance kept."),
    Typology("layering_chain", "Layering chain", "laundering",
             "A large amount moves through a straight chain of accounts one after another within hours, each "
             "account passing on almost all of it, before it leaves the bank.",
             "Sequential pass-through over several hops; each hop happens soon after the previous one and keeps "
             "almost the whole amount; intermediaries are otherwise quiet."),
    Typology("stack", "Stacked layers", "laundering",
             "Two or more layers of accounts where each account in one layer pays several accounts in the next "
             "layer, used to multiply the number of paths money takes.",
             "Groups of accounts paying groups of accounts in quick succession with similar amounts."),
    Typology("bipartite", "Bipartite transfer", "laundering",
             "A group of source accounts pays a group of destination accounts in a dense criss-cross pattern.",
             "Several senders each paying several of the same recipients within a short time."),
    Typology("payroll", "Payroll (innocent)", "innocent",
             "An employer pays salaries to a stable set of employees on the same day each month.",
             "Same recipients every month, amounts round and repeated each month, payments on the same date, "
             "inflows are regular client receipts; recipients spend the money normally afterwards."),
    Typology("wholesale", "Wholesale distribution (innocent)", "innocent",
             "A distributor receives many payments from a regular set of shops and pays its manufacturer or "
             "supplier a large share of it every few days.",
             "Recurring set of buyers over weeks, many small-to-medium payments, regular schedule of supplier "
             "payments, keeps a margin of 5-20%."),
    Typology("investment", "Investment or savings transfer (innocent)", "innocent",
             "A person receives a bonus, salary or sale proceeds and moves most of it to a broker, deposit or "
             "investment account within a day or two.",
             "One inflow from an employer or known client, one outflow to a single investment-type account the "
             "person has paid before or which many customers pay; no splitting."),
    Typology("property", "Property or large purchase (innocent)", "innocent",
             "A customer receives a loan disbursal and pays a builder or seller in a few large instalments, "
             "sometimes just under ₹10 lakh.",
             "Large loan credit first, then instalments to one seller spread over weeks; no rapid forwarding to "
             "many accounts."),
    Typology("family", "Family transfers (innocent)", "innocent",
             "Family members move money between each other's accounts, which can form a loop over days or weeks.",
             "Same small group of people, amounts that differ rather than shrinking by small fees, accounts have "
             "normal everyday spending around the transfers."),
]


# ---------------------------------------------------------------- embeddings
class HashEmbeddings:
    """Dependency-free fallback: hashed bag of words and word pairs (lexical, not semantic)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        words = re.findall(r"[a-z]+", text.lower())
        grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        v = np.zeros(self.dim, dtype="float32")
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dim] += 1 if (h >> 8) % 2 else -1
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text).tolist()


def get_embeddings():
    provider = os.getenv("EMBEDDINGS_PROVIDER", "auto").lower()
    if provider in ("auto", "gemini") and os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"))
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings  # optional dependency

        return HuggingFaceEmbeddings(model_name=os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    return HashEmbeddings()


# ---------------------------------------------------------------- index
class PatternLibrary:
    def __init__(self, embeddings=None, typologies: list[Typology] | None = None):
        self.typologies = typologies or TYPOLOGIES
        self.embeddings = embeddings or get_embeddings()
        texts = [f"{t.name}. {t.description} {t.tell_tale_checks}" for t in self.typologies]
        vecs = np.array(self.embeddings.embed_documents(texts), dtype="float32")
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = np.array([self.embeddings.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q)
        scores, idx = self.index.search(q, min(k, len(self.typologies)))
        out = []
        for score, i in zip(scores[0], idx[0]):
            t = self.typologies[int(i)]
            out.append({"id": t.id, "name": t.name, "kind": t.kind, "similarity": round(float(score), 3),
                        "description": t.description, "tell_tale_checks": t.tell_tale_checks})
        return out

    def get(self, typology_id: str) -> dict | None:
        for t in self.typologies:
            if t.id == typology_id:
                return {"id": t.id, "name": t.name, "kind": t.kind, "similarity": None,
                        "description": t.description, "tell_tale_checks": t.tell_tale_checks}
        return None

    def ids(self) -> list[str]:
        return [t.id for t in self.typologies]
