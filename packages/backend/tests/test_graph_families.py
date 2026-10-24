from types import SimpleNamespace

import pytest

from app.services.chat import retrieve_candidates
from app.services.graph_families import ASSERTED_FACT, ASSOCIATION, STRUCTURAL, family_of


def test_edge_types_map_to_families():
    assert family_of("MENTIONS") == STRUCTURAL
    assert family_of("CO_OCCURS_WITH") == ASSOCIATION
    assert family_of("ASSERTS") == ASSERTED_FACT


class _Embedder:
    async def embed_query(self, text):
        return [0.0, 0.0, 1.0]


class _NER:
    async def extract(self, text):
        return [SimpleNamespace(entity_type="ORGANIZATION", normalized_text="acme")]


class _DB:
    def __init__(self, facts):
        self.facts = facts

    async def search_similar_chunks(self, embedding, config_id, dim, limit):
        return [{"id": "sim-1", "distance": 0.1}, {"id": "fact-1", "distance": 0.9}]

    async def find_entities(self, pairs):
        return ["entity-acme"]

    async def get_asserted_fact_chunks(self, config_id, dim, entity_ids, embedding, limit):
        return self.facts

    async def get_chunks_mentioning_entities(self, config_id, dim, entity_ids, embedding, limit):
        return [{"id": "mention-1", "distance": 0.5}]


@pytest.mark.asyncio
async def test_asserted_facts_are_ranked_before_similarity_and_mentions():
    config = SimpleNamespace(id=1, embedding_dim=3)
    db = _DB(facts=[{"id": "fact-1", "distance": 0.9}])
    rows = await retrieve_candidates(
        "Did Acme acquire Beta?", db, _Embedder(), config, 10, True, _NER()
    )
    assert [row["id"] for row in rows] == ["fact-1", "sim-1", "mention-1"]


@pytest.mark.asyncio
async def test_without_facts_order_is_unchanged():
    config = SimpleNamespace(id=1, embedding_dim=3)
    db = _DB(facts=[])
    rows = await retrieve_candidates(
        "Did Acme acquire Beta?", db, _Embedder(), config, 10, True, _NER()
    )
    assert [row["id"] for row in rows] == ["sim-1", "mention-1", "fact-1"]
