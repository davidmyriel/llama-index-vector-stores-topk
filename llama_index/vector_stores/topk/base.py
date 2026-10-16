"""LlamaIndex vector store backed by TopK."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from topk_sdk import AsyncClient, Client
from topk_sdk.error import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    PartitionNotFoundError,
    QueryLsnTimeoutError,
)
from topk_sdk.query import LogicalExpr, field, fn, match, select, should
from topk_sdk.query import all as topk_all
from topk_sdk.schema import (
    f32_vector,
    keyword_index,
    semantic_index,
    text,
    vector_index,
)

from llama_index.vector_stores.topk._filters import (
    METADATA_FIELD,
    REF_DOC_ID_FIELD,
    translate_filters,
)

Metric = Literal["cosine", "dot_product", "euclidean"]

REGIONS = ("aws-us-east-1-elastica", "aws-eu-central-1-monstera", "gcp-us-east4-aloe")
TEXT_FIELD = "text"
VECTOR_FIELD = "vector"
METADATA_JSON_FIELD = "metadata_json"
_UPSERT_BATCH_SIZE = 200
_GET_NODES_LIMIT = 1000
_NOT_FOUND = (CollectionNotFoundError, PartitionNotFoundError)
_LSN_RETRIES = 3

_DENSE_MODES = {VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.MMR}
_KEYWORD_MODES = {VectorStoreQueryMode.TEXT_SEARCH, VectorStoreQueryMode.SPARSE}
_HYBRID_MODES = {VectorStoreQueryMode.HYBRID, VectorStoreQueryMode.SEMANTIC_HYBRID}


class TopKServerSideEmbedding(BaseEmbedding):
    """
    Placeholder embedding model for a ``TopKVectorStore`` that embeds on the server.

    ``VectorStoreIndex`` always resolves an embedding model, and falls back to
    OpenAI when you don't pass one. With server-side embeddings TopK embeds the
    text itself, so pass this instead: it calls no model and costs nothing, and
    the store ignores the placeholder vectors it returns.

    .. code-block:: python

        index = VectorStoreIndex.from_vector_store(store, embed_model=TopKServerSideEmbedding())
    """

    model_name: str = "topk-server-side"

    @classmethod
    def class_name(cls) -> str:
        return "TopKServerSideEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        return [0.0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return [0.0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return [0.0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


@dataclass
class _Plan:
    """A TopK query plus how to read scores back from its rows."""

    query: Any
    mode: str  # "dense", "keyword" or "hybrid"
    alpha: float
    lower_is_better: bool


class TopKVectorStore(BasePydanticVectorStore):
    """
    LlamaIndex vector store for `TopK <https://topk.io>`_, a search engine for AI agents.

    TopK runs keyword (BM25), semantic, vector and hybrid search, plus metadata
    filters, in one query against one index.

    Two ways to run it:

    * **Server-side embeddings** (``server_side_embeddings=True``, the default).
      TopK embeds the text itself with its ``semantic_index()``. You don't need an
      embedding model or an API key for one. Build the index with
      ``embed_model=TopKServerSideEmbedding()``.
    * **Bring your own embeddings** (``server_side_embeddings=False``). LlamaIndex
      computes vectors with your ``embed_model``; they are stored in an
      ``f32_vector`` field with a vector index, and the text also gets a BM25
      keyword index.

    Query modes (``vector_store_query_mode=`` on the retriever or query engine):

    * ``"default"``: dense search, with TopK's embeddings or yours
    * ``"text_search"`` (or ``"sparse"``): BM25 keyword search, for exact terms, codes and names
    * ``"hybrid"``: dense similarity and BM25 ranked together in one query; ``alpha``
      weights them (1 = dense only, 0 = BM25 only)
    * ``"mmr"``: maximal marginal relevance (bring-your-own embeddings only)

    Examples:
        .. code-block:: python

            from llama_index.core import Document, StorageContext, VectorStoreIndex
            from llama_index.vector_stores.topk import TopKServerSideEmbedding, TopKVectorStore

            store = TopKVectorStore(collection_name="docs")  # reads TOPK_API_KEY, TOPK_REGION
            index = VectorStoreIndex.from_documents(
                [Document(text="TopK runs hybrid search in one query.")],
                storage_context=StorageContext.from_defaults(vector_store=store),
                embed_model=TopKServerSideEmbedding(),
            )
            retriever = index.as_retriever(vector_store_query_mode="hybrid", alpha=0.5)
            retriever.retrieve("what is hybrid search?")
    """

    stores_text: bool = True
    flat_metadata: bool = False

    collection_name: str = Field(description="TopK collection to read and write.")
    partition: str | None = Field(default=None, description="Scope every read and write to this partition.")
    server_side_embeddings: bool = Field(default=True, description="Let TopK embed text on the server.")
    embedding_dimension: int | None = Field(default=None, description="Vector size for bring-your-own embeddings.")
    metric: Metric = Field(default="cosine", description="Vector metric for bring-your-own embeddings.")
    hybrid_alpha: float = Field(default=0.5, description="Default dense weight in hybrid search; BM25 gets 1 - alpha.")
    create_collection: bool = Field(default=True, description="Create the collection if missing.")

    _client: Client = PrivateAttr()
    _aclient: AsyncClient | None = PrivateAttr(default=None)
    _lsn: str | None = PrivateAttr(default=None)
    _collection_ready: bool = PrivateAttr(default=False)

    def __init__(
        self,
        collection_name: str = "llamaindex",
        *,
        server_side_embeddings: bool = True,
        client: Client | None = None,
        async_client: AsyncClient | None = None,
        api_key: str | None = None,
        region: str | None = None,
        host: str = "topk.io",
        https: bool = True,
        partition: str | None = None,
        metric: Metric = "cosine",
        embedding_dimension: int | None = None,
        hybrid_alpha: float = 0.5,
        create_collection: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Create the store.

        Args:
            collection_name: TopK collection to read and write. Created if missing.
            server_side_embeddings: Let TopK embed text. Set ``False`` to store the
                vectors your LlamaIndex ``embed_model`` computes.
            client: Existing ``topk_sdk.Client``. If given, ``api_key``/``region``/``host``/``https``
                are only used to create the async client.
            async_client: Existing ``topk_sdk.AsyncClient`` for the ``a*`` methods.
            api_key: TopK API key. Defaults to ``TOPK_API_KEY``. Create one at https://console.topk.io.
            region: TopK region, e.g. ``"aws-us-east-1-elastica"``. Defaults to ``TOPK_REGION``.
                See https://docs.topk.io/regions.
            host: TopK API host.
            https: Use HTTPS.
            partition: Scope every read and write to this partition (one per tenant or user).
            metric: Vector metric for bring-your-own embeddings.
            embedding_dimension: Vector size. Inferred from the first write if omitted.
            hybrid_alpha: Dense weight in hybrid search when the query doesn't set ``alpha``.
            create_collection: Create the collection if it doesn't exist.

        """
        if not 0.0 <= hybrid_alpha <= 1.0:
            msg = f"hybrid_alpha must be between 0 and 1, got {hybrid_alpha}"
            raise ValueError(msg)
        super().__init__(
            collection_name=collection_name,
            partition=partition,
            server_side_embeddings=server_side_embeddings,
            embedding_dimension=embedding_dimension,
            metric=metric,
            hybrid_alpha=hybrid_alpha,
            create_collection=create_collection,
            # TopK embeds the query itself, so the retriever shouldn't.
            is_embedding_query=not server_side_embeddings,
            **kwargs,
        )

        if client is None or async_client is None:
            api_key, region = _credentials(api_key, region, required=client is None)
        self._client = client or Client(api_key=api_key, region=region, host=host, https=https)
        if async_client is not None:
            self._aclient = async_client
        elif api_key and region:
            self._aclient = AsyncClient(api_key=api_key, region=region, host=host, https=https)

        if create_collection and (server_side_embeddings or embedding_dimension is not None):
            self._ensure_collection(embedding_dimension)

    @classmethod
    def class_name(cls) -> str:
        return "TopKVectorStore"

    @property
    def client(self) -> Client:
        """The underlying ``topk_sdk.Client``, for anything this wrapper doesn't cover."""
        return self._client

    @property
    def aclient(self) -> AsyncClient | None:
        """The underlying ``topk_sdk.AsyncClient``, if one is configured."""
        return self._aclient

    def with_partition(self, partition: str) -> TopKVectorStore:
        """
        Return a store scoped to another partition of the same collection.

        Partitions are fully isolated, so this keeps each tenant's or user's
        data separate without a ``tenant_id`` filter.
        """
        store = self.model_copy(update={"partition": partition})
        store._lsn = None
        return store

    # ------------------------------------------------------------------ setup

    def _schema(self, dimension: int | None) -> dict[str, Any]:
        if self.server_side_embeddings:
            # semantic_index() also builds the BM25 index used by text_search and hybrid.
            return {TEXT_FIELD: text().required().index(semantic_index())}
        if dimension is None:
            msg = "embedding_dimension is required to create a collection for bring-your-own embeddings"
            raise ValueError(msg)
        return {
            TEXT_FIELD: text().required().index(keyword_index()),
            VECTOR_FIELD: f32_vector(dimension=dimension).required().index(vector_index(metric=self.metric)),
        }

    def _ensure_collection(self, dimension: int | None) -> None:
        if self._collection_ready or not self.create_collection:
            return
        try:
            self._client.collections().create(self.collection_name, schema=self._schema(dimension))
        except CollectionAlreadyExistsError:
            pass
        self._collection_ready = True

    async def _aensure_collection(self, dimension: int | None) -> None:
        if self._collection_ready or not self.create_collection:
            return
        try:
            await self._aclient.collections().create(self.collection_name, schema=self._schema(dimension))
        except CollectionAlreadyExistsError:
            pass
        self._collection_ready = True

    def _collection(self) -> Any:
        return self._client.collection(self.collection_name, partition=self.partition)

    def _acollection(self) -> Any:
        return self._aclient.collection(self.collection_name, partition=self.partition)

    def _read_kwargs(self) -> dict[str, Any]:
        # Pass the last write's LSN so reads always see this store's own writes.
        return {"lsn": self._lsn} if self._lsn else {}

    def _read(self, method: str, *args: Any, **kwargs: Any) -> Any:
        # A read that waits for a fresh write's LSN can time out while TopK catches up; retry it.
        for attempt in range(_LSN_RETRIES):
            try:
                return getattr(self._collection(), method)(*args, **kwargs, **self._read_kwargs())
            except QueryLsnTimeoutError:
                if attempt == _LSN_RETRIES - 1:
                    raise
        return None

    async def _aread(self, method: str, *args: Any, **kwargs: Any) -> Any:
        for attempt in range(_LSN_RETRIES):
            try:
                return await getattr(self._acollection(), method)(*args, **kwargs, **self._read_kwargs())
            except QueryLsnTimeoutError:
                if attempt == _LSN_RETRIES - 1:
                    raise
        return None

    # ------------------------------------------------------------------ writes

    def _to_docs(self, nodes: Sequence[BaseNode]) -> list[dict[str, Any]]:
        docs = []
        for node in nodes:
            doc: dict[str, Any] = {
                "_id": node.node_id,
                TEXT_FIELD: node.get_content(metadata_mode=MetadataMode.NONE),
                REF_DOC_ID_FIELD: node.ref_doc_id or "",
                METADATA_FIELD: _filterable(node.metadata or {}),
                # The whole node minus text and embedding, so queries return it intact.
                METADATA_JSON_FIELD: json.dumps(
                    node_to_metadata_dict(node, remove_text=True, flat_metadata=False),
                    default=str,
                ),
            }
            if not self.server_side_embeddings:
                embedding = node.get_embedding()
                doc[VECTOR_FIELD] = [float(x) for x in embedding]
                self.embedding_dimension = self.embedding_dimension or len(embedding)
            docs.append(doc)
        return docs

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        """Upsert nodes. Existing nodes with the same id are replaced."""
        if not nodes:
            return []
        docs = self._to_docs(nodes)
        self._ensure_collection(self.embedding_dimension)
        collection = self._collection()
        for start in range(0, len(docs), _UPSERT_BATCH_SIZE):
            self._lsn = collection.upsert(docs[start : start + _UPSERT_BATCH_SIZE])
        return [node.node_id for node in nodes]

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        if self._aclient is None:
            return await asyncio.to_thread(self.add, nodes, **kwargs)
        if not nodes:
            return []
        docs = self._to_docs(nodes)
        await self._aensure_collection(self.embedding_dimension)
        collection = self._acollection()
        for start in range(0, len(docs), _UPSERT_BATCH_SIZE):
            self._lsn = await collection.upsert(docs[start : start + _UPSERT_BATCH_SIZE])
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete every node that came from the document ``ref_doc_id``."""
        self._delete(field(REF_DOC_ID_FIELD).eq(ref_doc_id))

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        await self._adelete(field(REF_DOC_ID_FIELD).eq(ref_doc_id))

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | LogicalExpr | None = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by id, by metadata filter, or both (nodes must match both)."""
        expr = _nodes_expr(node_ids, filters)
        if expr is not None:
            self._delete(expr)

    async def adelete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | LogicalExpr | None = None,
        **delete_kwargs: Any,
    ) -> None:
        expr = _nodes_expr(node_ids, filters)
        if expr is not None:
            await self._adelete(expr)

    def clear(self) -> None:
        """
        Delete every node in this store's partition (or the whole collection).

        Keeps the collection itself: deleting and recreating a TopK collection
        under the same name can drop the first writes after it.
        """
        self._delete(field("_id").is_not_null())

    async def aclear(self) -> None:
        await self._adelete(field("_id").is_not_null())

    def _delete(self, expr: LogicalExpr) -> None:
        try:
            self._lsn = self._collection().delete(expr)
        except _NOT_FOUND:
            pass

    async def _adelete(self, expr: LogicalExpr) -> None:
        if self._aclient is None:
            await asyncio.to_thread(self._delete, expr)
            return
        try:
            self._lsn = await self._acollection().delete(expr)
        except _NOT_FOUND:
            pass

    # ------------------------------------------------------------------ reads

    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | LogicalExpr | None = None,
    ) -> list[BaseNode]:
        """Fetch nodes by id and/or metadata filter (up to 1,000 per call)."""
        q = _get_nodes_query(node_ids, filters)
        try:
            rows = self._read("query", q)
        except _NOT_FOUND:
            return []
        return _ordered_nodes(rows, node_ids)

    async def aget_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | LogicalExpr | None = None,
    ) -> list[BaseNode]:
        if self._aclient is None:
            return await asyncio.to_thread(self.get_nodes, node_ids, filters)
        q = _get_nodes_query(node_ids, filters)
        try:
            rows = await self._aread("query", q)
        except _NOT_FOUND:
            return []
        return _ordered_nodes(rows, node_ids)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Run a dense, BM25, hybrid or MMR query. Higher similarities are better."""
        plan = self._plan(query, **kwargs)
        try:
            rows = self._read("query", plan.query)
        except _NOT_FOUND:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        if query.mode == VectorStoreQueryMode.MMR:
            vectors = self._read("get", [r["_id"] for r in rows], fields=[VECTOR_FIELD])
            return _mmr(query, rows, vectors, plan, **kwargs)
        return _to_result(rows, plan)

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if self._aclient is None:
            return await asyncio.to_thread(self.query, query, **kwargs)
        plan = self._plan(query, **kwargs)
        try:
            rows = await self._aread("query", plan.query)
        except _NOT_FOUND:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        if query.mode == VectorStoreQueryMode.MMR:
            vectors = await self._aread("get", [r["_id"] for r in rows], fields=[VECTOR_FIELD])
            return _mmr(query, rows, vectors, plan, **kwargs)
        return _to_result(rows, plan)

    def _plan(self, query: VectorStoreQuery, **kwargs: Any) -> _Plan:
        mode = query.mode
        alpha = self.hybrid_alpha if query.alpha is None else query.alpha
        if not 0.0 <= alpha <= 1.0:
            msg = f"alpha must be between 0 and 1, got {alpha}"
            raise ValueError(msg)
        k = query.similarity_top_k
        lower_is_better = not self.server_side_embeddings and self.metric == "euclidean"
        text_filter = None
        scores: dict[str, Any] = {}

        if mode in _KEYWORD_MODES:
            _require_query_str(query)
            scores["kw"] = fn.bm25_score()
            text_filter = match(query.query_str, field=TEXT_FIELD)
            rank, kind = field("kw"), "keyword"
        elif mode in _DENSE_MODES or mode in _HYBRID_MODES:
            if mode == VectorStoreQueryMode.MMR:
                if self.server_side_embeddings:
                    msg = "MMR needs stored vectors; use server_side_embeddings=False to use it"
                    raise ValueError(msg)
                k = kwargs.get("mmr_prefetch_k") or k * kwargs.get("mmr_prefetch_factor", 4)
            scores["dense"] = self._dense_score(query)
            rank, kind = (field("dense") * -1.0 if lower_is_better else field("dense")), "dense"
            if mode in _HYBRID_MODES:
                _require_query_str(query)
                scores["kw"] = fn.bm25_score()
                # should() scores keyword matches without dropping documents that have none.
                text_filter = should(query.query_str, field=TEXT_FIELD)
                rank = rank * alpha + field("kw") * (1.0 - alpha)
                kind = "hybrid"
                k = query.hybrid_top_k or k
        else:
            msg = f"Query mode {mode.value!r} isn't supported; use default, text_search, hybrid or mmr"
            raise ValueError(msg)

        q = select(TEXT_FIELD, METADATA_JSON_FIELD, **scores)
        if text_filter is not None:
            q = q.filter(text_filter)
        meta_filter = _query_filter(query)
        if meta_filter is not None:
            q = q.filter(meta_filter)
        q = q.sort(rank, asc=False).limit(k)
        return _Plan(query=q, mode=kind, alpha=alpha, lower_is_better=lower_is_better)

    def _dense_score(self, query: VectorStoreQuery) -> Any:
        if self.server_side_embeddings:
            _require_query_str(query)
            return fn.semantic_similarity(TEXT_FIELD, query.query_str)
        if query.query_embedding is None:
            msg = "This store uses your own embeddings, so the query needs query_embedding"
            raise ValueError(msg)
        return fn.vector_distance(VECTOR_FIELD, [float(x) for x in query.query_embedding])


def _credentials(api_key: str | None, region: str | None, required: bool) -> tuple:
    api_key = api_key or os.environ.get("TOPK_API_KEY")
    region = region or os.environ.get("TOPK_REGION")
    if required and not api_key:
        msg = "Set TOPK_API_KEY or pass api_key=... (create a key at https://console.topk.io)"
        raise ValueError(msg)
    if required and not region:
        msg = f"Set TOPK_REGION or pass region=... (one of {', '.join(REGIONS)}; see https://docs.topk.io/regions)"
        raise ValueError(msg)
    return api_key, region


def _require_query_str(query: VectorStoreQuery) -> None:
    if not query.query_str:
        msg = f"Query mode {query.mode.value!r} needs query_str"
        raise ValueError(msg)


def _query_filter(query: VectorStoreQuery) -> LogicalExpr | None:
    clauses = []
    meta = translate_filters(query.filters)
    if meta is not None:
        clauses.append(meta)
    if query.doc_ids:
        clauses.append(field(REF_DOC_ID_FIELD).in_(list(query.doc_ids)))
    if query.node_ids:
        clauses.append(field("_id").in_(list(query.node_ids)))
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else topk_all(clauses)


def _nodes_expr(node_ids: list[str] | None, filters: MetadataFilters | LogicalExpr | None) -> LogicalExpr | None:
    clauses = []
    if node_ids:
        clauses.append(field("_id").in_(list(node_ids)))
    meta = translate_filters(filters)
    if meta is not None:
        clauses.append(meta)
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else topk_all(clauses)


def _get_nodes_query(node_ids: list[str] | None, filters: MetadataFilters | LogicalExpr | None) -> Any:
    expr = _nodes_expr(node_ids, filters)
    if expr is None:
        msg = "get_nodes needs node_ids or filters"
        raise ValueError(msg)
    return select(TEXT_FIELD, METADATA_JSON_FIELD).filter(expr).limit(_GET_NODES_LIMIT)


def _to_node(row: dict[str, Any]) -> BaseNode:
    node = metadata_dict_to_node(json.loads(row[METADATA_JSON_FIELD]), text=row.get(TEXT_FIELD, ""))
    node.id_ = row["_id"]
    return node


def _ordered_nodes(rows: list[dict[str, Any]], node_ids: list[str] | None) -> list[BaseNode]:
    nodes = [_to_node(r) for r in rows]
    if node_ids:
        order = {node_id: i for i, node_id in enumerate(node_ids)}
        nodes.sort(key=lambda n: order.get(n.node_id, len(order)))
    return nodes


def _score(row: dict[str, Any], plan: _Plan) -> float:
    dense = row.get("dense") or 0.0
    if plan.lower_is_better:
        dense = -dense
    kw = row.get("kw") or 0.0
    if plan.mode == "keyword":
        return float(kw)
    if plan.mode == "hybrid":
        return float(dense * plan.alpha + kw * (1.0 - plan.alpha))
    return float(dense)


def _to_result(rows: list[dict[str, Any]], plan: _Plan) -> VectorStoreQueryResult:
    return VectorStoreQueryResult(
        nodes=[_to_node(r) for r in rows],
        similarities=[_score(r, plan) for r in rows],
        ids=[r["_id"] for r in rows],
    )


def _mmr(
    query: VectorStoreQuery,
    rows: list[dict[str, Any]],
    vectors: dict[str, dict[str, Any]],
    plan: _Plan,
    **kwargs: Any,
) -> VectorStoreQueryResult:
    # TopK doesn't return indexed vectors from queries, so they're fetched by id.
    rows = [r for r in rows if r["_id"] in vectors]
    if not rows:
        return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    similarities, ids = get_top_k_mmr_embeddings(
        query.query_embedding,
        [vectors[r["_id"]][VECTOR_FIELD] for r in rows],
        similarity_top_k=query.similarity_top_k,
        embedding_ids=[r["_id"] for r in rows],
        mmr_threshold=query.mmr_threshold or kwargs.get("mmr_threshold"),
    )
    by_id = {r["_id"]: r for r in rows}
    return VectorStoreQueryResult(
        nodes=[_to_node(by_id[i]) for i in ids],
        similarities=list(similarities),
        ids=list(ids),
    )


def _filterable(value: Any) -> Any:
    """Keep the metadata values TopK can store and filter on; everything survives in metadata_json."""
    if isinstance(value, dict):
        out = {str(k): _filterable(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v is not None}
    if isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        items = [v for v in value if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        if items and len(items) == len(value) and len({type(v) for v in items}) == 1:
            return list(items)
        return None
    return None
