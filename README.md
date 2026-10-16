# llama-index-vector-stores-topk

**LlamaIndex vector store and hybrid search for [TopK](https://topk.io).** Keyword (BM25), semantic, vector and hybrid search, metadata filters, and isolated per-tenant partitions, all in one query against one index. TopK can also embed text on the server, so you don't need an embedding model.

```bash
pip install llama-index-vector-stores-topk
```

```python
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.topk import TopKServerSideEmbedding, TopKVectorStore

store = TopKVectorStore(collection_name="docs")  # TopK embeds text for you
index = VectorStoreIndex.from_documents(
    [
        Document(text="Error code E-4021 means the payment gateway timed out.", metadata={"source": "runbook"}),
        Document(text="Quarterly revenue grew 12% thanks to cloud sales.", metadata={"source": "report"}),
    ],
    storage_context=StorageContext.from_defaults(vector_store=store),
    embed_model=TopKServerSideEmbedding(),  # no embedding model is called
)

retriever = index.as_retriever(vector_store_query_mode="hybrid", similarity_top_k=3)
retriever.retrieve("why did checkout fail?")
```

Set `TOPK_API_KEY` (create one at [console.topk.io](https://console.topk.io)) and `TOPK_REGION` (for example `aws-us-east-1-elastica`, see [regions](https://docs.topk.io/regions)).

## What is TopK?

TopK is a **hybrid retrieval engine** for AI applications and agents. In a typical RAG stack, an embedding service, a vector database, a keyword engine and a reranker are separate pieces glued together in application code. TopK runs all of them in one system:

| | Typical vector-database setup | TopK |
|---|---|---|
| **Embeddings** | You run an embedding pipeline and store the vectors | `semantic_index()` embeds on the server ([semantic search](https://docs.topk.io/guides/semantic-search)) |
| **Hybrid search** | Vector and keyword searches run separately, then get merged (often with Reciprocal Rank Fusion) | One query over one index, so the top results really are the top results ([true hybrid search](https://docs.topk.io/guides/true-hybrid-search)) |
| **Retrieval types** | Dense vectors, plus keyword search bolted on | Dense, sparse and multi-vector search, BM25, regex, filters and custom scoring in one query |
| **Storage** | Memory- or disk-bound nodes you size and scale | Object storage with memory and NVMe caching ([architecture](https://docs.topk.io/architecture)), which TopK says costs 10x less |
| **Multi-tenancy** | A `tenant_id` filter | Fully isolated partitions per tenant ([concepts](https://docs.topk.io/concepts#multi-tenancy)) |

TopK publishes these figures: 1B+ documents per partition, under 100ms p99 query latency at 1B documents, and under 1 second data freshness ([architecture](https://docs.topk.io/architecture)). Retrieval benchmarks are at [topk.io/benchmarks](https://www.topk.io/benchmarks).

## Why use TopK with LlamaIndex?

- **No embedding model to run.** With `server_side_embeddings=True` (the default), TopK embeds nodes and queries on the server. That means no OpenAI embeddings bill and no extra embedding API key. Pass `embed_model=TopKServerSideEmbedding()` so LlamaIndex doesn't fall back to OpenAI.
- **Keyword search that actually filters.** Agents look up exact strings like error codes, SKUs, function names and ticket IDs. Vector similarity ranks *something* first even when nothing matches; BM25 returns nothing. `vector_store_query_mode="text_search"` gives you that.
- **True hybrid search in one call.** `vector_store_query_mode="hybrid"` scores dense similarity and BM25 together on the server, weighted by `alpha`. Nodes with no shared keyword still rank on meaning, and keyword matches get boosted. You don't need `QueryFusionRetriever` or an in-memory `BM25Retriever`.
- **Agent memory per user or tenant.** `store.with_partition("user-42")` gives each user a fully isolated partition with no cross-tenant filter to forget.
- **It's a standard LlamaIndex vector store.** `TopKVectorStore` is a `BasePydanticVectorStore`, so it works with `VectorStoreIndex`, retrievers, query engines, chat engines, `IngestionPipeline`, `delete_ref_doc`, `MetadataFilters` and async. Nodes come back with their text, metadata and relationships intact.

## Query modes

Set `vector_store_query_mode` on `as_retriever()`, `as_query_engine()` or `as_chat_engine()`.

| `vector_store_query_mode` | What it does | Use it for |
|---|---|---|
| `"default"` | Dense search: TopK's built-in embeddings, or your `embed_model` with `server_side_embeddings=False` | Natural-language questions |
| `"text_search"` (or `"sparse"`) | BM25 full-text; returns only nodes that share a term | Exact codes, names, identifiers |
| `"hybrid"` | Dense + BM25 in one query; `alpha` weights them (1 = dense only, 0 = BM25 only, default 0.5) | General RAG and agent retrieval |
| `"mmr"` | Maximal marginal relevance over your own vectors | Diverse results (bring-your-own embeddings only) |

```python
index.as_retriever(vector_store_query_mode="text_search").retrieve("E-4021")  # BM25 only
index.as_retriever(vector_store_query_mode="hybrid", alpha=0.7).retrieve("payment gateway timeout")  # 70% dense
index.as_query_engine(vector_store_query_mode="hybrid", similarity_top_k=5).query("Why did checkout fail?")
```

Dense and BM25 scores sit on different scales. Look at a few `NodeWithScore.score` values from a hybrid retriever before you tune `alpha`.

### Bring your own embeddings

```python
from llama_index.embeddings.openai import OpenAIEmbedding

store = TopKVectorStore(
    collection_name="docs-openai",
    server_side_embeddings=False,
    metric="cosine",  # or "dot_product", "euclidean"
)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(vector_store=store),
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
)
index.as_retriever(vector_store_query_mode="mmr", similarity_top_k=4).retrieve("diverse results please")
```

LlamaIndex computes the vectors with your `embed_model`; the collection is created with the right dimension on the first write. The text field also gets a BM25 index, so `"text_search"` and `"hybrid"` work the same way (hybrid combines your vectors with BM25).

## Metadata filters

Pass standard LlamaIndex `MetadataFilters` to any retriever or query engine:

```python
from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
        MetadataFilter(key="tags", value=["payments", "billing"], operator=FilterOperator.ANY),
        MetadataFilter(key="author.name", value="Ada"),  # nested metadata uses dots
    ],
    condition=FilterCondition.AND,
)
index.as_retriever(filters=filters, vector_store_query_mode="hybrid").retrieve("outage")
```

Operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `nin`, `contains`, `any`, `all`, `text_match` (substring) and `is_empty`, combined with `AND`, `OR`, `NOT` and nested `MetadataFilters`. `doc_id`, `ref_doc_id` and `document_id` match the source document.

You can also pass a native TopK expression for anything else TopK supports:

```python
from topk_sdk.query import field

index.as_retriever(filters=field("metadata.year").in_([2024, 2025]) & field("metadata.title").regexp_match("^RFC"))
```

## Multi-tenant agent memory with partitions

```python
memories = TopKVectorStore(collection_name="agent-memory")
alice = memories.with_partition("user-alice")
bob = memories.with_partition("user-bob")


def index_for(store):
    return VectorStoreIndex.from_vector_store(store, embed_model=TopKServerSideEmbedding())


index_for(alice).insert(Document(text="Alice prefers window seats and vegetarian meals."))
index_for(bob).insert(Document(text="Bob is allergic to peanuts."))

index_for(alice).as_retriever(vector_store_query_mode="hybrid").retrieve("food preferences")  # never sees Bob's data
```

A partition is created on its first write. Reading a partition that was never written returns nothing.

## Updates and deletes

```python
index.delete_ref_doc("doc-123")  # every node from one source document
store.delete_nodes(node_ids=["node-1", "node-2"])  # specific nodes
store.delete_nodes(filters=MetadataFilters(filters=[MetadataFilter(key="source", value="stale-import")]))
store.get_nodes(filters=...)  # fetch nodes by id or filter
store.clear()  # empty the collection (or the partition)
```

Re-adding a node with the same id replaces it, so re-running an `IngestionPipeline` updates nodes in place.

## Use with agents

For agents, install [`llama-index-tools-topk`](https://github.com/topk-io/llama-index-tools-topk). It turns this store into `keyword_search`, `semantic_search` and `hybrid_search` tools whose descriptions tell the agent which one fits (an exact error code goes to BM25, a paraphrased question goes to semantic search). It also adds `remember` / `recall` long-term memory in a partition per user, and a `TopKMemoryBlock` for LlamaIndex `Memory`.

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.topk import TopKToolSpec

tools = TopKToolSpec(store, metadata_fields={"source": "'runbook' or 'report'"}).to_tool_list()
agent = FunctionAgent(tools=tools, llm=llm)
```

## Good to know

- **Reads see your own writes.** The store passes the LSN of its last write to every read, so a query right after `insert` includes the new nodes.
- **Upserts replace whole nodes.** Adding a node with an existing id overwrites it.
- **Use a new collection name after deleting one.** Writes right after recreating a just-deleted collection under the same name can be lost. `store.clear()` empties a collection without deleting it.
- **Metadata** comes back exactly as you stored it. Values TopK can't filter on (for example mixed-type lists or custom objects) are still returned, just not filterable.
- **Async.** `aquery`, `async_add`, `adelete`, `adelete_nodes`, `aget_nodes` and `aclear` use TopK's native async client.
- **Scores.** Higher is better in every mode. With `metric="euclidean"` the score is the negative distance.

## FAQ

**Does TopK work with LlamaIndex?** Yes. `llama-index-vector-stores-topk` provides `TopKVectorStore`, a standard LlamaIndex vector store for `VectorStoreIndex`, retrievers and query engines.

**Can I do hybrid search (BM25 + vectors) in LlamaIndex without a fusion retriever?** Yes. `vector_store_query_mode="hybrid"` ranks dense similarity and BM25 together in a single TopK query. You don't need `QueryFusionRetriever`, Reciprocal Rank Fusion, or a separate in-memory BM25 index.

**Do I need OpenAI or another embedding provider?** No. With server-side embeddings, TopK embeds nodes and queries itself; pass `embed_model=TopKServerSideEmbedding()` to your index. You can still bring any LlamaIndex embedding model with `server_side_embeddings=False`.

**How do I isolate data per user or tenant?** Use partitions: `store.with_partition("tenant-id")`. Each partition is fully isolated and scales independently.

**Why isn't this in the llama_index monorepo?** LlamaIndex no longer accepts new integration packages in its repository; new integrations are published to PyPI on their own. This one follows the same `llama_index.vector_stores.*` namespace, so imports look like any other vector store.

**Is TopK open source?** The SDKs ([topk-io/topk](https://github.com/topk-io/topk)) and this integration are MIT-licensed. TopK itself is a managed service ([topk.io](https://topk.io)), and you can also deploy it in your own VPC.

## Development

```bash
uv venv && uv pip install -e . --group test
pytest tests/unit_tests
TOPK_API_KEY=... TOPK_REGION=aws-us-east-1-elastica pytest tests/integration_tests  # runs against live TopK
```

## Links

- TopK docs: https://docs.topk.io
- Agent tools and memory: https://github.com/topk-io/llama-index-tools-topk
- TopK Python SDK: https://pypi.org/project/topk-sdk/
- Coding agents: `npx skills add topk-io/topk --skill topk` installs the TopK Agent Skill
