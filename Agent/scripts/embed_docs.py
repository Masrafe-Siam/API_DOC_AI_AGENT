import os
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
CHUNK_COLLECTION_NAME = "api_docs_chunks"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL_NAME)
client = PersistentClient(path=CHROMA_PERSIST_DIR)

def get_embedding(text: str) -> List[float]:
    return embedder.encode(text, show_progress_bar=False).tolist()

def store_chunks_in_chroma(chunks: List[str]):
    if CHUNK_COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(name=CHUNK_COLLECTION_NAME)

    collection = client.get_or_create_collection(name=CHUNK_COLLECTION_NAME)

    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk-{idx}"],
            metadatas=[{"chunk_index": idx}]
        )
        print(f"Stored chunk {idx + 1}/{len(chunks)}")

    print("All chunks stored in ChromaDB!")

def embed_docs(chunks: List[str]):
    print(f"Embedding {len(chunks)} chunks...")
    store_chunks_in_chroma(chunks)
    print("Collection created:", [c.name for c in client.list_collections()])

if __name__ == "__main__":
    from chunk_docs import chunk_docs
    chunks = chunk_docs()
    embed_docs(chunks)
