from src.retrieval.qdrant_db import get_qdrant_client
from src.ingestion.embedder import get_embedder, embed_texts

COLLECTION_NAME = "ragwatch_demo"

def search(query: str, client=None, model=None, top_k: int = 5):
    if client is None:
        client = get_qdrant_client()
    if model is None:
        model = get_embedder()

    query_vector = embed_texts(model, [query])[0]

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=top_k,
    )
    return results


if __name__ == "__main__":
    query = "What is the attention mechanism in transformers?"
    results = search(query)

    print(f"Query: {query}\n")
    for i, hit in enumerate(results, start=1):
        print(f"--- Result {i} (score: {hit.score:.4f}) ---")
        print(f"Page: {hit.payload.get('page')}")
        print(hit.payload.get("text")[:200])
        print()