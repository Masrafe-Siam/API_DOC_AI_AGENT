import os
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
CHUNK_COLLECTION_NAME = "api_docs_chunks"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TOP_K = 5

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_collection(name=CHUNK_COLLECTION_NAME)

def embed_text(text: str) -> List[float]:
    return embedding_model.encode(text).tolist()

def retrieve_relevant_chunks(query: str, collection, top_k: int = TOP_K) -> List[str]:
    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]

def build_prompt(question: str, chunks: List[str]) -> str:
    context = "\n\n".join(chunks)
    return f"""You are a helpful documentation assistant.

Use ONLY the context below to answer the question. If the answer is not in the context, say: "The documentation does not contain enough information to answer that."

---

CONTEXT:
{context}

---

QUESTION:
{question}

---

ANSWER:"""

def get_answer_from_ollama(prompt: str) -> str:
    response = ollama.chat(model=OLLAMA_MODEL, messages=[
        {"role": "system", "content": "You are a helpful documentation assistant."},
        {"role": "user", "content": prompt}
    ])
    return response["message"]["content"].strip()

def answer_question(question: str):
    collection = load_chroma()
    chunks = retrieve_relevant_chunks(question, collection)
    prompt = build_prompt(question, chunks)
    answer = get_answer_from_ollama(prompt)
    return answer, chunks

# --- Streamlit UI ---
def run_streamlit_ui():
    import streamlit as st

    st.title("API Docs Q&A Assistant (Ollama + ChromaDB)")
    question = st.text_input("Ask a question about the API docs:", placeholder="e.g. What is a PaymentIntent?")
    if st.button("Get Answer") and question:
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            collection = load_chroma()
            chunks = retrieve_relevant_chunks(question, collection)
            prompt = build_prompt(question, chunks)
            answer = get_answer_from_ollama(prompt)
            st.subheader("Answer:")
            st.markdown(answer)
            with st.expander("Show Context Chunks"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}:**\n```text\n{chunk}\n```")

if __name__ == "__main__":
    run_streamlit_ui()
