# this ties retrieval and generation into one callable RAG function

import os
from dotenv import load_dotenv
load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ragwatch")

from src.retrieval.qdrant_db import get_qdrant_client
from src.ingestion.embedder import get_embedder
from src.retrieval.search import search
from src.generation.generator import get_groq_client, generate_answer


def rag_query(query: str, top_k: int = 5):
    qdrant_client = get_qdrant_client()
    embed_model = get_embedder()
    groq_client = get_groq_client()

    results = search(query, client=qdrant_client, model=embed_model, top_k=top_k)
    context_chunks = [hit.payload["text"] for hit in results]

    answer = generate_answer(groq_client, query, context_chunks)
    return answer, results

if __name__ == "__main__":
    query = "What is the attention mechanism in transformers?"
    answer, results = rag_query(query)

    print(f"Query: {query}\n")
    print(f"Answer:\n{answer}\n")
    print("--- Sources ---")
    for hit in results:
        print(f"Page {hit.payload.get('page')} (score: {hit.score:.4f})")
