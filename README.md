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
# Enter you OPENAI key within ""
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

## 10. Run the agent

```bash
# go to the scripts directory
cd scripts
```
```bash
# run the main.py in terminal
#For cli interface
python main.py
```
```bash
#For streamlit Web UI
streamlit run main.py
```

### Pervious Version

```bash
# if you are in the scripts directory go back to Agent directory 
#if you are in Agent directory SKIP this
cd ..
```
```bash
# go to old_arch directory
cd old_arch
```
```bash
# run the chunk_loader.py to load and chunk the documents
python chunk_loder.py
```
```bash
# run the embed_store.py to create chunk and load it to chromaDB
python embed_store.py
```
```bash
# run the qa_agent.py for Question and Answer
python qa_agent.py
```
```bash
# run the qa_agent_ollama.py for local llm model Ollama
python qa_agent_ollama.py
```
```bash
# run the qa_web_app.py for Streamlit web ul using ollama
python qa_web_app.py
```
```bash
# run the qa_agent_openai.py for openai llm model
# warning: wont work if you use "OPENAI_MODEL_NAME= gpt-3.5-turbo"
# to use this you will need "OPENAI_MODEL_NAME=gpt-4o" and a 'OPENAI_API_KEY = "sk-proj-...."'
# if you have these set this into .env file and then run
python qa_agent_openai.py
```

## API documentation

-For this project i used stripe API documentation https://docs.stripe.com/api .

-Firstly i save this document as .html file in the docs directory and build API_DOC_AI_AGENT with the document.

** You can use any type of documentation (.html or .md) file.

## How the app is working

### Step 1 : Data Preparation

--store the API documention file in docs directory

--Support .html and .md file type (extendable)

--load the document

--chunking the document into samller group

### Step 2 : Embedding & Storage

--used Embedding Model SentenceTransformer (e.g., all-MiniLM-L6-v2)

--store those chunks into the Vector Database (used ChromaDB)

### Step 3 : Q&A Loop

--local llm model for question and answer season (used ollama)

--in old arch there is also openai llm model but needed gpt-4o for that

--streamlit for web ui interaction

1.Accept natural language question from user

2.Embed the question and perform semantic search in the vector DB

3.Retrieve top-k relevant documentation chunks

4.Construct a prompt with user question + retrieved context

5.Send prompt to LLM and return answer

# ✨ Example Queries

--How do I authenticate API requests?

--What parameters are needed for the create_charge endpoint?

--Is rate limiting applied per user or per account?

# 🧩 Design Choices

-Vector DB = ChromaDB: Lightweight, Python-native vector database optimized for local storage and fast similarity search in RAG setups.

-Embedding model = SentenceTransformer: Generates high-quality dense embeddings from text using transformer models, ideal for semantic search. (e.g., all-MiniLM-L6-v2)

-LLM = Ollama: Runs large language models like LLaMA locally with minimal setup, ensuring privacy and offline capability.

-Frontend = Streamlit: Simple and fast way to build interactive web apps in Python, ideal for prototyping and user-friendly interfaces.

-CLI support: Allows command-line interaction for power users and automation, enabling flexible access to core features without a GUI.

# 🧱 Tech Stack
Component	Choice(s)
Language	Python
Vector DB	ChromaDB 
Embedding	Sentence Transformers (all-MiniLM-L6-v2)
LLM	OpenAI GPT-3.5/GPT-4, local models via Ollama
Interface	CLI (default), Streamlit

# 📬 Contact
MASRAFE BIN HANNAN SIAM

📧 Personal     : masrafesiammbhs1633@gmail.com

📧 Institutional : siam35-1022@diu.edu.bd

[🔗 LinkedIn Profile](https://www.linkedin.com/in/masrafe-siam-44108b202/)

[🔗 GitHub Profile](https://github.com/Masrafe-Siam)

