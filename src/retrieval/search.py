from src.retrieval.qdrant_db import get_qdrant_client
from src.ingestion.embedder import get_embedder, embed_texts

COLLECTION_NAME = "ragwatch_demo"

def search(query: str, top_k: int = 5):
    client = get_qdrant_client()
    model = get_embedder()

    query_vector = embed_texts(model, [query])[0]
    # This returns a list of vectors, we take the first one since we only have one query

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=top_k
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