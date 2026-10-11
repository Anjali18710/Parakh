"""Case storage: MongoDB when MONGODB_URI is set, otherwise in memory."""
from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from typing import Protocol


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CaseRepository(Protocol):
    def insert(self, collection: str, doc: dict) -> None: ...
    def update(self, collection: str, key: str, fields: dict) -> None: ...
    def get(self, collection: str, key: str) -> dict | None: ...
    def list(self, collection: str, limit: int = 200) -> list[dict]: ...


class MemoryRepository:
    def __init__(self):
        self._data: dict[str, dict[str, dict]] = {}
        self._lock = threading.Lock()

    def insert(self, collection, doc):
        with self._lock:
            self._data.setdefault(collection, {})[doc["_id"]] = copy.deepcopy(doc)

    def update(self, collection, key, fields):
        with self._lock:
            self._data.setdefault(collection, {}).setdefault(key, {"_id": key}).update(copy.deepcopy(fields))

    def get(self, collection, key):
        with self._lock:
            doc = self._data.get(collection, {}).get(key)
            return copy.deepcopy(doc) if doc else None

    def list(self, collection, limit=200):
        with self._lock:
            docs = sorted(self._data.get(collection, {}).values(), key=lambda d: d.get("created_at", ""), reverse=True)
            return copy.deepcopy(docs[:limit])


class MongoRepository:
    def __init__(self, uri: str, db: str, client=None):
        if client is None:
            from pymongo import MongoClient

            client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        self.db = client[db]

    def insert(self, collection, doc):
        self.db[collection].insert_one(copy.deepcopy(doc))

    def update(self, collection, key, fields):
        self.db[collection].update_one({"_id": key}, {"$set": fields}, upsert=True)

    def get(self, collection, key):
        return self.db[collection].find_one({"_id": key})

    def list(self, collection, limit=200):
        return list(self.db[collection].find().sort("created_at", -1).limit(limit))


def make_repository(uri: str | None, db: str) -> CaseRepository:
    return MongoRepository(uri, db) if uri else MemoryRepository()
