from qdrant_client.models import Distance, VectorParams, PointStruct
from torch import chunk
from src.retrieval.qdrant_db import get_qdrant_client
from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import get_embedder, embed_texts
from pathlib import Path
import uuid

COLLECTION_NAME = "ragwatch_demo"
VECTOR_SIZE = 384  # dimension of the embedding model

def create_collection(client, collection_name: str, vector_size: int):
    # recreate_collection drops and rebuilds the collection if it exists
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

def upsert_chunks(client, collection_name: str, chunks, vectors):
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={
                "text": chunk.page_content,
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    pdf_files = list((project_root / "data" / "raw").glob("*.pdf"))

    docs = load_pdf(str(pdf_files[0]))
    chunks = chunk_documents(docs)

    model = get_embedder()
    vectors = embed_texts(model, [c.page_content for c in chunks])

    client = get_qdrant_client()
    create_collection(client, COLLECTION_NAME, VECTOR_SIZE)
    upsert_chunks(client, COLLECTION_NAME, chunks, vectors)

    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' now has {info.points_count} points")