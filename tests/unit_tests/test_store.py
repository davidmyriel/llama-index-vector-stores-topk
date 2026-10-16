import pytest
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from llama_index.vector_stores.topk import TopKServerSideEmbedding, TopKVectorStore, translate_filters
from llama_index.vector_stores.topk.base import _filterable


class FakeClient:
    def collections(self):
        raise AssertionError("should not create a collection")


def test_is_a_llamaindex_vector_store():
    assert issubclass(TopKVectorStore, BasePydanticVectorStore)
    assert TopKVectorStore.class_name() == "TopKVectorStore"


def test_requires_credentials(monkeypatch):
    monkeypatch.delenv("TOPK_API_KEY", raising=False)
    monkeypatch.delenv("TOPK_REGION", raising=False)
    with pytest.raises(ValueError, match="TOPK_API_KEY"):
        TopKVectorStore()
    monkeypatch.setenv("TOPK_API_KEY", "x")
    with pytest.raises(ValueError, match="TOPK_REGION"):
        TopKVectorStore()


def test_alpha_validation():
    with pytest.raises(ValueError, match="hybrid_alpha"):
        TopKVectorStore(client=FakeClient(), hybrid_alpha=2, create_collection=False)
    store = TopKVectorStore(client=FakeClient(), create_collection=False)
    with pytest.raises(ValueError, match="alpha"):
        store._plan(VectorStoreQuery(query_str="q", mode=VectorStoreQueryMode.HYBRID, alpha=1.5))


def test_server_side_mode_skips_query_embedding():
    assert TopKVectorStore(client=FakeClient(), create_collection=False).is_embedding_query is False
    assert TopKVectorStore(client=FakeClient(), server_side_embeddings=False).is_embedding_query is True


def test_byo_embeddings_create_lazily():
    # No embedding_dimension yet, so the collection waits for the first add().
    store = TopKVectorStore(client=FakeClient(), server_side_embeddings=False)
    assert store.embedding_dimension is None


def test_query_validation():
    semantic = TopKVectorStore(client=FakeClient(), create_collection=False)
    byo = TopKVectorStore(client=FakeClient(), server_side_embeddings=False)
    with pytest.raises(ValueError, match="query_str"):
        semantic._plan(VectorStoreQuery(query_embedding=[0.1], mode=VectorStoreQueryMode.DEFAULT))
    with pytest.raises(ValueError, match="query_embedding"):
        byo._plan(VectorStoreQuery(query_str="q", mode=VectorStoreQueryMode.DEFAULT))
    with pytest.raises(ValueError, match="MMR"):
        semantic._plan(VectorStoreQuery(query_str="q", mode=VectorStoreQueryMode.MMR))
    with pytest.raises(ValueError, match="svm"):
        semantic._plan(VectorStoreQuery(query_str="q", mode=VectorStoreQueryMode.SVM))


def test_placeholder_embedding_calls_no_model():
    emb = TopKServerSideEmbedding()
    assert emb.get_text_embedding_batch(["a", "b"]) == [[0.0], [0.0]]
    assert emb.get_query_embedding("q") == [0.0]


def test_filterable_metadata():
    assert _filterable({"a": 1, "b": None, "c": [1, 2], "d": [1, "x"], "e": {"f": "g"}, "o": object()}) == {
        "a": 1,
        "c": [1, 2],
        "e": {"f": "g"},
    }


def test_translate_filters_shapes():
    assert translate_filters(None) is None
    assert translate_filters(MetadataFilters(filters=[])) is None
    for op, value in [
        (FilterOperator.EQ, "a"),
        (FilterOperator.NE, "a"),
        (FilterOperator.GT, 1),
        (FilterOperator.GTE, 1.5),
        (FilterOperator.LT, 1),
        (FilterOperator.LTE, 1),
        (FilterOperator.IN, ["a", "b"]),
        (FilterOperator.NIN, [1, 2]),
        (FilterOperator.CONTAINS, "a"),
        (FilterOperator.ANY, ["a", "b"]),
        (FilterOperator.ALL, ["a", "b"]),
        (FilterOperator.TEXT_MATCH, "sub"),
        (FilterOperator.IS_EMPTY, None),
    ]:
        assert (
            translate_filters(MetadataFilters(filters=[MetadataFilter(key="k", value=value, operator=op)])) is not None
        )
    nested = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1),
            MetadataFilters(filters=[MetadataFilter(key="b", value=2)], condition=FilterCondition.NOT),
        ],
        condition=FilterCondition.OR,
    )
    assert translate_filters(nested) is not None


def test_translate_filters_rejects_unsupported():
    with pytest.raises(ValueError, match="text_match_insensitive"):
        translate_filters(
            MetadataFilters(
                filters=[MetadataFilter(key="k", value="x", operator=FilterOperator.TEXT_MATCH_INSENSITIVE)]
            )
        )
    with pytest.raises(ValueError, match="non-empty list"):
        translate_filters(MetadataFilters(filters=[MetadataFilter(key="k", value="x", operator=FilterOperator.IN)]))
    with pytest.raises(TypeError):
        translate_filters({"k": "v"})


def test_reads_retry_lsn_timeouts():
    from topk_sdk.error import QueryLsnTimeoutError

    calls = []

    class Flaky:
        def query(self, q, **kwargs):
            calls.append(kwargs)
            if len(calls) < 3:
                raise QueryLsnTimeoutError("QueryLsnTimeout")
            return []

    class Client:
        def collection(self, name, partition=None):
            return Flaky()

    store = TopKVectorStore(client=Client(), create_collection=False)
    store._lsn = "42"
    assert store._read("query", "q") == []
    assert calls == [{"lsn": "42"}] * 3

    calls.clear()
    Flaky.query = lambda self, q, **kw: (_ for _ in ()).throw(QueryLsnTimeoutError("x"))
    import pytest as _pytest

    with _pytest.raises(QueryLsnTimeoutError):
        store._read("query", "q")
