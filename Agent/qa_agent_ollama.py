import os
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests

# --- Load environment variables ---
load_dotenv()

# --- Constants ---
CHUNK_COLLECTION_NAME = "api_docs_chunks"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")  # e.g., llama3, mistral, codellama
TOP_K = 5  # Top relevant chunks to retrieve

# --- Load embedding model ---
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Initialize ChromaDB client ---
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_collection(name=CHUNK_COLLECTION_NAME)

# --- Text embedding function ---
def embed_text(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()

# --- Retrieve top relevant chunks ---
def retrieve_relevant_chunks(query: str, top_k: int = TOP_K) -> List[str]:
    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]

# --- Prompt builder ---
def build_prompt(question: str, chunks: List[str]) -> str:
    context = "\n\n".join(chunks)
    prompt = f"""You are a documentation assistant for a developer API.

Use ONLY the following documentation context to answer the user's question.

If the context is not sufficient to answer, say "The documentation does not contain enough information to answer that."

---

CONTEXT:
{context}

---

QUESTION:
{question}

---

ANSWER:"""
    return prompt

# --- Local LLM call (Ollama) ---
def get_answer_from_llm(prompt: str, model: str = OLLAMA_MODEL) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error generating response from Ollama: {e}"

# --- Main Q&A function ---
def answer_query(question: str):
    print("\nRetrieving relevant documentation chunks...")
    chunks = retrieve_relevant_chunks(question)

    print("\nContext:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---\n{chunk}")

    print("\nGenerating answer using Ollama...")
    prompt = build_prompt(question, chunks)
    answer = get_answer_from_llm(prompt)

    print("\nAnswer:")
    print(answer)

# --- CLI Loop ---
if __name__ == "__main__":
    print("\nAPI Docs Q&A Agent (w/ Ollama LLM)\n")
    while True:
        user_question = input("Ask a question (or type 'exit'): ").strip()
        if user_question.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        answer_query(user_question)
