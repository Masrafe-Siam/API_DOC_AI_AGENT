import os
from typing import List
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from dotenv import load_dotenv

load_dotenv()

# Settings
CHUNK_COLLECTION_NAME = "api_docs_chunks"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
TOP_K = 3

# Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Use PersistentClient (not Client!)
client = PersistentClient(path=CHROMA_PERSIST_DIR)

# Check collection exists
available_collections = [c.name for c in client.list_collections()]
print("Available collections:", available_collections)

if CHUNK_COLLECTION_NAME not in available_collections:
    raise ValueError(f"Collection '{CHUNK_COLLECTION_NAME}' not found. Run embed_store.py first.")

collection = client.get_collection(name=CHUNK_COLLECTION_NAME)

# Embed query
def embed_text(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()

# Retrieve top chunks
def retrieve_relevant_chunks(query: str, top_k: int = TOP_K):
    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]

# QA loop
def answer_query(question: str):
    print("\nRetrieving relevant context...")
    context_chunks = retrieve_relevant_chunks(question)

    print("\nTop relevant documentation chunks:")
    for i, chunk in enumerate(context_chunks):
        print(f"\n--- Chunk {i+1} ---\n{chunk}")

    print("\n[TODO] You can now feed these to an LLM to answer based on them.")

if __name__ == "__main__":
    print("API Docs Q&A Agent\n")
    while True:
        user_question = input("Ask a question (or type 'exit'): ").strip()
        if user_question.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        answer_query(user_question)
