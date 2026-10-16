"""Semantic, keyword and hybrid search over one TopK collection, with no embedding model.

Run: TOPK_API_KEY=... TOPK_REGION=aws-us-east-1-elastica python examples/quickstart.py
"""

import uuid

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from llama_index.vector_stores.topk import TopKServerSideEmbedding, TopKVectorStore

store = TopKVectorStore(collection_name=f"llamaindex-quickstart-{uuid.uuid4().hex[:6]}")
index = VectorStoreIndex.from_documents(
    [
        Document(
            text="Error code E-4021 means the payment gateway timed out. Retry after 30 seconds.",
            metadata={"source": "runbook"},
        ),
        Document(text="Quarterly revenue grew 12% thanks to cloud sales.", metadata={"source": "report"}),
        Document(
            text="Earnings increased this quarter because of strong subscription growth.", metadata={"source": "report"}
        ),
        Document(text="Restart the ingestion worker if the queue backs up.", metadata={"source": "runbook"}),
    ],
    storage_context=StorageContext.from_defaults(vector_store=store),
    embed_model=TopKServerSideEmbedding(),
)

QUERIES = [("text_search", "E-4021"), ("default", "how did revenue change?"), ("hybrid", "checkout payment failed")]
for mode, query in QUERIES:
    print(f"\n{mode}: {query!r}")
    for hit in index.as_retriever(vector_store_query_mode=mode, similarity_top_k=2).retrieve(query):
        print(f"  {hit.score:6.3f}  [{hit.node.metadata['source']}] {hit.node.get_content()}")

runbooks = MetadataFilters(filters=[MetadataFilter(key="source", value="runbook")])
print("\nfiltered:", [h.node.get_content() for h in index.as_retriever(filters=runbooks).retrieve("anything")])
store.client.collections().delete(store.collection_name)
