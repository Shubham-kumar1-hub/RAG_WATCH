from sentence_transformers import SentenceTransformer


# small (384-dimension) open-source embedding model
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

def embed_texts(model: SentenceTransformer, texts: list[str]):
    return model.encode(texts, normalize_embeddings=True)

# normalize_embeddings=True ->  scales every vector to unit length.
# This matters because it lets us use cosine similarity correctly and consistently when we configure Qdrant's distance metric


if __name__ == "__main__":
    model = get_embedder()
    sample = ["Attention is all you need."]
    vectors = embed_texts(model, sample)

    print(f"Model: {MODEL_NAME}")
    print(f"Vector dimension: {vectors.shape[1]}")
    print(f"First 5 values: {vectors[0][:5]}")
