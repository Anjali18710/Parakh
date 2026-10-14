"""Test-only shim: mongomock predates the `sort` argument newer pymongo passes to bulk updates.

Production uses the real pymongo driver against MongoDB; this only lets tests use mongomock in memory.
"""
import mongomock
from mongomock.collection import BulkOperationBuilder

_original = BulkOperationBuilder.add_update


def _add_update(self, *args, sort=None, **kwargs):  # noqa: ARG001 - mongomock has no sort support
    return _original(self, *args, **kwargs)


BulkOperationBuilder.add_update = _add_update
MongoClient = mongomock.MongoClient
