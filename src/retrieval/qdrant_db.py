import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()  # Load environment variables from .env file

def get_qdrant_client():
    url = os.getenv("QDRANT_URL")
    return QdrantClient(url=url)

if __name__ == "__main__":
    client = get_qdrant_client()
    collections = client.get_collections()   # Simple test to check if the connection is successful
    print("Connected to Qdrant successfully")
    print("Existing collections:", collections)