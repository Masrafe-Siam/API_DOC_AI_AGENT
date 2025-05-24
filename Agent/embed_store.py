import os
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# --- Embedding model ---
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight, fast & good quality

# --- ChromaDB collection name ---
CHUNK_COLLECTION_NAME = "api_docs_chunks"

# --- Initialize sentence-transformers model ---
print(f"Loading sentence-transformers model '{EMBED_MODEL_NAME}'...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# --- Initialize ChromaDB client ---
client = chromadb.Client(Settings(
    persist_directory="chroma_store",  # local storage directory
    anonymized_telemetry=False
))

def get_embedding(text: str) -> List[float]:
    # SentenceTransformer's encode returns a numpy array; convert to list
    embedding = embedder.encode(text, show_progress_bar=False)
    return embedding.tolist()

def store_chunks_in_chroma(chunks: List[str]):
    # Delete old collection if exists
    existing_collections = [c.name for c in client.list_collections()]
    if CHUNK_COLLECTION_NAME in existing_collections:
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
        print(f"✅ Stored chunk {idx+1}/{len(chunks)}")

    print("📦 All chunks stored in ChromaDB!")

if __name__ == "__main__":
    # --- Example chunk loader functions ---
    def load_document(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Simple sliding window chunking with overlap.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    # --- Load your document ---
    DOCS_PATH = "D:\\Masrafe\\Coding\\Git_Hub_code\\ml_project\\Api_Doc_Ai_agent\\Agent\\docs\\Stripe API Reference.html"  # Can be .md or .html
    raw_text = load_document(DOCS_PATH)
    chunks = chunk_text(raw_text)
    print(f"Document chunked into {len(chunks)} pieces.")

    # --- Store chunks and embeddings in ChromaDB ---
    store_chunks_in_chroma(chunks)
    print("✅ All chunks processed and stored in ChromaDB.")