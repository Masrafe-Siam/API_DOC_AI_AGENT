import os
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from openai import OpenAI

load_dotenv()

# --- Constants ---
CHUNK_COLLECTION_NAME = "api_docs_chunks"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"  # or gpt-4 ifhave access
TOP_K = 5  # Number of top chunks to retrieve

# --- Load embedding model ---
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Initialize ChromaDB client ---
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_collection(name=CHUNK_COLLECTION_NAME)

# --- Initialize OpenAI client ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Embedding function ---
def embed_text(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()

# --- Retrieve top relevant chunks ---
def retrieve_relevant_chunks(query: str, top_k: int = TOP_K):
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]  # List[str]

# --- Create a system prompt for the LLM ---
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

# --- LLM call ---
def get_answer_from_llm(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# --- Core QA loop ---
def answer_query(question: str):
    print("\nRetrieving relevant documentation chunks...")
    chunks = retrieve_relevant_chunks(question)

    print("\nContext:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---\n{chunk}")

    print("\nGenerating answer using LLM...")
    prompt = build_prompt(question, chunks)
    answer = get_answer_from_llm(prompt)

    print("\nAnswer:")
    print(answer)

# --- CLI ---
if __name__ == "__main__":
    print("API Docs Q&A Agent (w/ OpenAI LLM)\n")
    while True:
        user_question = input("Ask a question (or type 'exit'): ").strip()
        if user_question.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        answer_query(user_question)
