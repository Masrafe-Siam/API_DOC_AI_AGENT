# API_DOC_AI_AGENT

Objective: Develop a functional prototype of an Al agent that can answer natural
language questions about a specific API, using its official documentation as the
knowledge base. The agent must leverage vector database embeddings to retrieve
relevant context from the documentation before generating an answer with a Large
Language Model (LLM).

Background: Navigating extensive AP! documentation can be time-consuming.
Developers often need quick answers to specific questions about endpoints, parameters,
authentication, or usage examples. This project aims to build an intelligent agent that
understands a user's question, finds the most relevant sections in the API documentation
using semantic search (via vector embeddings), and then uses an LLM to synthesize a clear
and concise answer based only on that retrieved context. This is a practical application of
the Retrieval-Augmented Generation (RAG) pattern with an agentic decision-making
component (deciding what context is relevant).

## Project Structure

```
.
├── Agent/
│   ├── __pycache__/
│   ├── chroma_store/          # Vector store directory (Chroma DB)
│   ├── docs/                  # Uploaded documents for processing
│   ├── old_arch/              # Older versions of architecture
│   │   ├── chunk_loader.py
│   │   ├── embed_store.py
│   │   ├── qa_agent_ollama.py
│   │   ├── qa_agent_openai.py
│   │   ├── qa_agent.py
│   │   ├── qa_web_app.py
│
├── scripts/                   # Main scripts for document processing
│   ├── __pycache__/
│   ├── chunk_docs.py          # Splits documents into chunks
│   ├── embed_docs.py          # Embeds document chunks into vector store
│   ├── load_docs.py           # Loads documents from directory
│   ├── main.py                # Main entry point for the pipeline
│   ├── query_agent.py         # Handles querying the embedded data
│
├── .env                       # Environment variable configuration
├── requirements.txt           # Python dependencies
├── .gitignore
├── README.md                  # Project documentation (this file)

```

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/Masrafe-Siam/API_DOC_AI_AGENT
```

## 2. Go to the Agent Directory

```bash
cd Agent
```

## 3. Set Up Environment

```bash
python -m venv venv

# windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Make Directory For Docuuments and Vector DB storage

```bash
# for .html / .md documents files
mkdir docs
```
```bash
# for chromaDB storage
mkdir chroma_store
```

## 6. Create Environment Variables File

```bash
touch .env
```

## 7 . Add Environment Variables

```bash
# chage the key if you have gpt-4 "OPENAI_MODEL_NAME=gpt-4o"
echo "OPENAI_MODEL_NAME= gpt-3.5-turbo" >> .env
# for local llm model
echo "OLLAMA_MODEL= llama3" >> .env  
# Enter you OPENAI key
echo 'OPENAI_API_KEY = "sk-proj-...."' >> .env  
# paste your path for the saved document file within ""
echo 'DOCS_PATH = "..."' >> .env  
# paste your choraDB storage directory path within ""
echo 'CHROMA_PERSIST_DIR = "..."' >> .env 
```

## 8. Prepare your API documentation files

Place your API Documents file (.html / .md) inside the docs directory

## 9. Local LLM Model

Download and Install https://ollama.com/

```bash
# open cmd terminal & activate ollama
ollama run llama3.2
```

