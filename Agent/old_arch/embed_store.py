import os
from typing import List
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from dotenv import load_dotenv

load_dotenv()

# Constants
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_COLLECTION_NAME = "api_docs_chunks"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")

# Load model
print(f"Loading model: {EMBED_MODEL_NAME}")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Initialize ChromaDB PersistentClient
client = PersistentClient(path=CHROMA_PERSIST_DIR)

def get_embedding(text: str) -> List[float]:
    return embedder.encode(text, show_progress_bar=False).tolist()

def store_chunks_in_chroma(chunks: List[str]):
    # Delete old collection if exists
    if CHUNK_COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(name=CHUNK_COLLECTION_NAME)

    collection = client.create_collection(name=CHUNK_COLLECTION_NAME)

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

if __name__ == "__main__":
    def load_document(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    DOCS_PATH = os.getenv("DOCS_PATH")
    raw_text = load_document(DOCS_PATH)
    chunks = chunk_text(raw_text)

    print(f"Document chunked into {len(chunks)} pieces.")
    store_chunks_in_chroma(chunks)
    print("Collection created:", [c.name for c in client.list_collections()])
