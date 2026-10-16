import hashlib
import os
import re
import uuid

import pytest
from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from topk_sdk import AsyncClient, Client

DOCS = [
    Document(
        doc_id="rev",
        text="Quarterly revenue grew 12% thanks to cloud sales.",
        metadata={"source": "report", "year": 2025, "tags": ["finance"], "author": {"name": "Ada"}},
    ),
    Document(
        doc_id="e4021",
        text="Error code E-4021 means the payment gateway timed out.",
        metadata={"source": "runbook", "year": 2024, "tags": ["payments", "errors"], "author": {"name": "Lin"}},
    ),
    Document(
        doc_id="earn",
        text="Earnings increased this quarter because of strong subscription growth.",
        metadata={"source": "report", "year": 2024, "tags": ["finance"], "author": {"name": "Ada"}},
    ),
    Document(
        doc_id="restart",
        text="Restart the ingestion worker if the queue backs up.",
        metadata={"source": "runbook", "year": 2023, "tags": ["ops"], "owner": "sre-team"},
    ),
]


class BagOfWordsEmbedding(BaseEmbedding):
    """Deterministic test embeddings: texts that share words get similar vectors."""

    dim: int = 64

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture(scope="session")
def creds() -> dict:
    if not os.environ.get("TOPK_API_KEY"):
        pytest.skip("TOPK_API_KEY not set")
    return {"api_key": os.environ["TOPK_API_KEY"], "region": os.environ.get("TOPK_REGION", "aws-us-east-1-elastica")}


@pytest.fixture(scope="session")
def topk_client(creds) -> Client:
    return Client(**creds)


@pytest.fixture
def async_client(creds) -> AsyncClient:
    return AsyncClient(**creds)


@pytest.fixture
def collection_name(topk_client: Client):
    # A fresh name per test: TopK doesn't support reusing a just-deleted collection name.
    name = f"li_test_{uuid.uuid4().hex[:10]}"
    yield name
    try:
        topk_client.collections().delete(name)
    except Exception:
        pass
