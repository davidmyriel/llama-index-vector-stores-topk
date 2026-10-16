"""Live tests against TopK: server-side embeddings, BM25, hybrid, filters, partitions, deletes, MMR, async."""

import pytest
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from topk_sdk.query import field

from llama_index.vector_stores.topk import TopKServerSideEmbedding, TopKVectorStore

from .conftest import DOCS, BagOfWordsEmbedding

pytestmark = pytest.mark.integration


def build(store, docs=DOCS, embed_model=None):
    return VectorStoreIndex.from_documents(
        docs,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=embed_model or TopKServerSideEmbedding(),
    )


@pytest.fixture
def semantic_store(topk_client, collection_name) -> TopKVectorStore:
    store = TopKVectorStore(collection_name=collection_name, client=topk_client)
    build(store)
    return store


@pytest.fixture
def semantic_index(semantic_store) -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(semantic_store, embed_model=TopKServerSideEmbedding())


@pytest.fixture
def byo_index(topk_client, collection_name) -> VectorStoreIndex:
    store = TopKVectorStore(collection_name=collection_name, client=topk_client, server_side_embeddings=False)
    return build(store, embed_model=BagOfWordsEmbedding())


def doc_ids(nodes):
    return [n.node.ref_doc_id for n in nodes]


def meta(**kwargs):
    return MetadataFilters(filters=[MetadataFilter(key=k, value=v) for k, v in kwargs.items()])


# ---------------------------------------------------------------- server-side embeddings


def test_semantic_search_needs_no_embedding_model(semantic_index):
    nodes = semantic_index.as_retriever(similarity_top_k=2).retrieve("how did revenue change?")
    assert set(doc_ids(nodes)) == {"rev", "earn"}
    assert nodes[0].score is not None and nodes[0].score >= nodes[1].score


def test_nodes_round_trip(semantic_index):
    node = semantic_index.as_retriever(similarity_top_k=1).retrieve("payment gateway timeout")[0].node
    original = DOCS[1]
    assert node.ref_doc_id == "e4021"
    assert node.get_content() == original.text
    assert node.metadata == original.metadata


def test_text_search_finds_exact_codes(semantic_index):
    retriever = semantic_index.as_retriever(vector_store_query_mode="text_search", similarity_top_k=4)
    assert doc_ids(retriever.retrieve("E-4021")) == ["e4021"]


def test_text_search_only_returns_keyword_matches(semantic_index):
    retriever = semantic_index.as_retriever(vector_store_query_mode="text_search", similarity_top_k=4)
    assert retriever.retrieve("zebra") == []


def test_hybrid_keeps_documents_without_keyword_matches(semantic_index):
    # should() scores BM25 matches without filtering out everything else.
    retriever = semantic_index.as_retriever(vector_store_query_mode="hybrid", similarity_top_k=4)
    nodes = retriever.retrieve("E-4021 earnings")
    assert len(nodes) == 4
    assert doc_ids(nodes)[0] == "e4021"


def test_hybrid_alpha_moves_ranking(semantic_store):
    def top(alpha):
        q = VectorStoreQuery(
            query_str="subscription", similarity_top_k=4, mode=VectorStoreQueryMode.HYBRID, alpha=alpha
        )
        return semantic_store.query(q).ids

    keyword_only = top(0.0)
    # Only "earn" contains "subscription"; with alpha=0 (BM25 only) it must lead.
    assert keyword_only[0] == semantic_store.get_nodes(filters=meta(doc_id="earn"))[0].node_id
    assert len(top(1.0)) == 4


def test_hybrid_top_k(semantic_store):
    q = VectorStoreQuery(query_str="revenue", similarity_top_k=4, hybrid_top_k=1, mode=VectorStoreQueryMode.HYBRID)
    assert len(semantic_store.query(q).ids) == 1


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (meta(source="report"), {"rev", "earn"}),
        (
            MetadataFilters(filters=[MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE)]),
            {"rev", "e4021", "earn"},
        ),
        (MetadataFilters(filters=[MetadataFilter(key="year", value=2024, operator=FilterOperator.LT)]), {"restart"}),
        (
            MetadataFilters(filters=[MetadataFilter(key="source", value="report", operator=FilterOperator.NE)]),
            {"e4021", "restart"},
        ),
        (
            MetadataFilters(filters=[MetadataFilter(key="year", value=[2023, 2025], operator=FilterOperator.IN)]),
            {"rev", "restart"},
        ),
        (
            MetadataFilters(filters=[MetadataFilter(key="year", value=[2023, 2025], operator=FilterOperator.NIN)]),
            {"e4021", "earn"},
        ),
        (
            MetadataFilters(filters=[MetadataFilter(key="tags", value="errors", operator=FilterOperator.CONTAINS)]),
            {"e4021"},
        ),
        (
            MetadataFilters(filters=[MetadataFilter(key="tags", value=["ops", "errors"], operator=FilterOperator.ANY)]),
            {"e4021", "restart"},
        ),
        (
            MetadataFilters(
                filters=[MetadataFilter(key="tags", value=["payments", "errors"], operator=FilterOperator.ALL)]
            ),
            {"e4021"},
        ),
        (
            MetadataFilters(filters=[MetadataFilter(key="owner", value="sre", operator=FilterOperator.TEXT_MATCH)]),
            {"restart"},
        ),
        (
            MetadataFilters(filters=[MetadataFilter(key="owner", value=None, operator=FilterOperator.IS_EMPTY)]),
            {"rev", "e4021", "earn"},
        ),
        (meta(**{"author.name": "Ada"}), {"rev", "earn"}),
        (meta(doc_id="restart"), {"restart"}),
        (
            MetadataFilters(
                filters=[MetadataFilter(key="year", value=2025), MetadataFilter(key="year", value=2023)],
                condition=FilterCondition.OR,
            ),
            {"rev", "restart"},
        ),
        (
            MetadataFilters(filters=[MetadataFilter(key="source", value="report")], condition=FilterCondition.NOT),
            {"e4021", "restart"},
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(key="source", value="report"),
                    MetadataFilters(filters=[MetadataFilter(key="year", value=2025)], condition=FilterCondition.NOT),
                ]
            ),
            {"earn"},
        ),
        (field("metadata.year").eq(2023), {"restart"}),
    ],
)
def test_filters(semantic_index, filters, expected):
    nodes = semantic_index.as_retriever(similarity_top_k=4, filters=filters).retrieve("anything")
    assert set(doc_ids(nodes)) == expected


def test_filters_apply_to_text_search(semantic_index):
    retriever = semantic_index.as_retriever(
        vector_store_query_mode="text_search", similarity_top_k=4, filters=meta(source="runbook")
    )
    assert doc_ids(retriever.retrieve("quarter revenue")) == []


def test_doc_ids_restrict_retrieval(semantic_index):
    nodes = semantic_index.as_retriever(similarity_top_k=4, doc_ids=["rev", "restart"]).retrieve("revenue")
    assert set(doc_ids(nodes)) == {"rev", "restart"}


def test_get_nodes(semantic_store):
    nodes = semantic_store.get_nodes(filters=meta(source="runbook"))
    assert {n.ref_doc_id for n in nodes} == {"e4021", "restart"}
    ids = [n.node_id for n in nodes]
    assert [n.node_id for n in semantic_store.get_nodes(node_ids=ids[::-1])] == ids[::-1]
    with pytest.raises(ValueError):
        semantic_store.get_nodes()


def test_delete_by_ref_doc_id(semantic_index, semantic_store):
    semantic_index.delete_ref_doc("e4021")
    retriever = semantic_index.as_retriever(vector_store_query_mode="text_search", similarity_top_k=4)
    assert retriever.retrieve("E-4021") == []
    assert len(semantic_store.get_nodes(filters=meta(source="runbook"))) == 1


def test_delete_nodes_and_clear(semantic_store):
    runbook = [n.node_id for n in semantic_store.get_nodes(filters=meta(source="runbook"))]
    semantic_store.delete_nodes(node_ids=runbook[:1])
    assert len(semantic_store.get_nodes(filters=meta(source="runbook"))) == 1
    semantic_store.delete_nodes(filters=meta(source="report"))
    assert semantic_store.get_nodes(filters=meta(source="report")) == []
    semantic_store.clear()
    q = VectorStoreQuery(query_str="anything", similarity_top_k=4)
    assert semantic_store.query(q).ids == []


def test_upsert_replaces_by_node_id(semantic_store):
    node = semantic_store.get_nodes(filters=meta(doc_id="restart"))[0]
    node.set_content("Restart the billing worker when invoices stall.")
    semantic_store.add([node])
    retriever = VectorStoreIndex.from_vector_store(semantic_store, embed_model=TopKServerSideEmbedding()).as_retriever(
        vector_store_query_mode="text_search", similarity_top_k=4
    )
    assert doc_ids(retriever.retrieve("invoices")) == ["restart"]
    assert retriever.retrieve("ingestion") == []


def test_partitions_isolate_tenants(topk_client, collection_name):
    store = TopKVectorStore(collection_name=collection_name, client=topk_client)
    alice, bob = store.with_partition("alice"), store.with_partition("bob")
    build(alice, [Document(doc_id="a", text="Alice prefers dark mode.")])
    build(bob, [Document(doc_id="b", text="Bob prefers light mode.")])

    def ask(s):
        index = VectorStoreIndex.from_vector_store(s, embed_model=TopKServerSideEmbedding())
        return doc_ids(index.as_retriever(similarity_top_k=4).retrieve("which mode does the user prefer?"))

    assert ask(alice) == ["a"]
    assert ask(bob) == ["b"]
    assert ask(store.with_partition("carol")) == []


def test_missing_collection_reads_empty(topk_client, collection_name):
    store = TopKVectorStore(collection_name=collection_name, client=topk_client, create_collection=False)
    assert store.query(VectorStoreQuery(query_str="x", similarity_top_k=2)).ids == []
    assert store.get_nodes(node_ids=["x"]) == []
    store.delete("x")


def test_long_text_is_searchable(topk_client, collection_name):
    store = TopKVectorStore(collection_name=collection_name, client=topk_client)
    long_text = ("Background filler sentence about nothing in particular. " * 120) + "The secret codename is BLUEHERON."
    assert len(long_text) > 4096
    store.add([TextNode(text=long_text, id_="long-node")])
    q = VectorStoreQuery(query_str="BLUEHERON", similarity_top_k=1, mode=VectorStoreQueryMode.TEXT_SEARCH)
    assert len(store.query(q).ids) == 1


# ---------------------------------------------------------------- bring your own embeddings


def test_byo_vector_search(byo_index):
    nodes = byo_index.as_retriever(similarity_top_k=1).retrieve("payment gateway error code")
    assert doc_ids(nodes) == ["e4021"]


def test_byo_hybrid_and_text_search(byo_index):
    hybrid = byo_index.as_retriever(vector_store_query_mode="hybrid", similarity_top_k=4, alpha=0.3)
    assert doc_ids(hybrid.retrieve("E-4021"))[0] == "e4021"
    text_search = byo_index.as_retriever(vector_store_query_mode="text_search", similarity_top_k=4)
    assert doc_ids(text_search.retrieve("ingestion")) == ["restart"]


def test_byo_mmr(byo_index):
    retriever = byo_index.as_retriever(
        vector_store_query_mode="mmr", similarity_top_k=2, vector_store_kwargs={"mmr_threshold": 0.3}
    )
    assert len(retriever.retrieve("quarterly revenue earnings")) == 2


def test_byo_euclidean(topk_client, collection_name):
    store = TopKVectorStore(
        collection_name=collection_name, client=topk_client, server_side_embeddings=False, metric="euclidean"
    )
    index = build(store, embed_model=BagOfWordsEmbedding())
    nodes = index.as_retriever(similarity_top_k=4).retrieve("payment gateway error code")
    assert doc_ids(nodes)[0] == "e4021"
    scores = [n.score for n in nodes]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------- async


async def test_async_round_trip(topk_client, async_client, collection_name):
    store = TopKVectorStore(collection_name=collection_name, client=topk_client, async_client=async_client)
    index = VectorStoreIndex.from_vector_store(store, embed_model=TopKServerSideEmbedding(), use_async=True)
    for doc in DOCS:
        await index.ainsert(doc)

    retriever = index.as_retriever(vector_store_query_mode="hybrid", similarity_top_k=2)
    assert doc_ids(await retriever.aretrieve("E-4021"))[0] == "e4021"
    assert {n.ref_doc_id for n in await store.aget_nodes(filters=meta(source="report"))} == {"rev", "earn"}

    await store.adelete("e4021")
    text_search = index.as_retriever(vector_store_query_mode="text_search", similarity_top_k=4)
    assert await text_search.aretrieve("E-4021") == []

    await store.adelete_nodes(filters=meta(source="report"))
    assert await store.aget_nodes(filters=meta(source="report")) == []
    await store.aclear()
    assert (await store.aquery(VectorStoreQuery(query_str="x", similarity_top_k=4))).ids == []
