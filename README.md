---
title: AI Code Assistant
emoji: 💻
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 💻 AI Code Assistant: Graph & RAG-Powered Codebase Intelligence

An advanced, production-grade, full-stack AI Codebase Intelligence and Debugging Assistant. Powered by **Retrieval-Augmented Generation (RAG)**, it enables deep semantic scanning, Abstract Syntax Tree (AST) parsing, hybrid retrieval, graph-enhanced context expansion, and a multi-turn streaming chat dashboard.

Built using **FastAPI** for high-performance backend serving (with WebSockets for streaming) and **Streamlit** for a responsive, interactive UI. Optimized to run locally or deployed instantly to **Hugging Face Spaces** via Docker.

---

## 🚀 Key Features

* **📦 Multi-Turn Streaming Chat:** Real-time token streaming over WebSockets for a snappy, conversational experience.
* **🧠 Graph-Enhanced Context Expansion:** Traverses structural code relationships (like imports, inheritance, and caller-callee graphs) to construct high-fidelity prompt context.
* **🔍 Hybrid Search Retrieval:** Balances lexical relevance (BM25 keyword matching) and deep semantic embedding similarity (vector search) for precise code retrieval.
* **🧬 Code-Aware AST Parsing:** Leverages Abstract Syntax Trees (AST) to build structural representations of code files (identifying functions, classes, imports) with a robust line-based fallback chunker for non-supported languages.
* **⚡ Dual LLM Provider Support:** Integrates natively with high-performance inference via **Groq** (`llama-3.3-70b-versatile`) and **Gemini API** (`gemini-1.5-pro` / `gemini-1.5-flash`).
* **🐳 One-Command Containerization:** Fully Dockerized setup using `start.sh` to run backend and frontend concurrently in a single container.

---

## 📐 System Architecture

```text
    ┌────────────────────────┐
    │  Streamlit Frontend   │◄────────────────────────┐
    │     (Port 7860)        │                         │
    └───────────┬────────────┘                         │
                │ WebSockets (Stream) / HTTP           │
                ▼                                      │
    ┌────────────────────────┐                         │
    │    FastAPI Backend     ├────────────────┐        │
    │     (Port 8000)        │                │        │
    └───────────┬────────────┘                │        │
                │                             ▼        ▼
                │ Ingestion          ┌───────────────────┐
                ▼                    │    LLM Provider   │
    ┌────────────────────────┐       │  (Groq / Gemini)  │
    │  AST Parser & Chunker  │       └───────────────────┘
    └───────────┬────────────┘
                │
                ▼
    ┌────────────────────────┐
    │    Hybrid Retriever    │
    ├───────────┬────────────┤
    │  Chroma   │    BM25    │
    │  (Vector) │  (Lexical) │
    └───────────┴────────────┘
```

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit, WebSockets (via `websocket-client`)
* **Backend:** FastAPI, Uvicorn, Pydantic v2, Python-dotenv
* **RAG Pipeline:** ChromaDB (Vector Store), PyTorch, Hugging Face Transformers (`BAAI/bge-base-en-v1.5` embeddings & `BAAI/bge-reranker-base` reranker)
* **LLM Engine:** Groq SDK, Google Generative AI (Gemini) SDK
* **Deployment:** Docker, Bash shell automation

---

## 📂 Project Structure

```text
├── backend/
│   ├── app/
│   │   ├── api/             # API Endpoints (Ingest, Query, Repo management)
│   │   ├── generation/      # LLM clients, prompt engineering, memory management
│   │   ├── ingestion/       # AST Parser, chunkers, embedding creators, graph builders
│   │   ├── models/          # Schemas and DB Models
│   │   ├── retrieval/       # Hybrid retriever, graph expansion, rerankers, query analyzers
│   │   ├── utils/           # Logging & utility helper modules
│   │   └── main.py          # FastAPI application entrypoint
│   ├── requirements.txt     # Python backend dependencies
│   ├── .env.example         # Template for environment configurations
│   └── data/                # Local database, repos, and graphs (Git ignored)
├── frontend/
│   ├── app.py               # Streamlit application UI
│   └── requirements.txt     # Python frontend dependencies
├── Dockerfile               # Build configuration for Hugging Face Spaces & Production
├── start.sh                 # Multi-service bash runner script
└── README.md                # Project documentation
```

---

## ⚙️ Environment Configuration

Before running the application, configure your credentials. Create a `.env` file inside the `backend` folder:

```bash
# Navigate to backend and create .env
cp backend/.env.example backend/.env
```

Open `backend/.env` and update the keys:
```env
# LLM Provider selection (groq, gemini, or openai)
LLM_PROVIDER=groq

# Enter your API Keys (only the active provider's key is mandatory)
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# Model Configuration
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
```

---

## 💻 Local Setup & Development

### 1. Prerequisites
Ensure you have **Python 3.10+** and **Git** installed on your system.

### 2. Backend Installation
```bash
# Navigate to backend
cd backend

# Create a virtual environment
python -m venv venv
source venv/Scripts/activate     # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API documentation will be available at `http://localhost:8000/docs`.

### 3. Frontend Installation
In a new terminal window:
```bash
# Navigate to frontend
cd frontend

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/Scripts/activate     # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the Streamlit application
streamlit run app.py --server.port=8501
```
Open `http://localhost:8501` to use the dashboard!

---

## 🐳 Running with Docker

You can run the entire stack locally in a single Docker container mimicking the Hugging Face Space environment.

```bash
# Build the Docker image
docker build -t ai-code-assistant .

# Run the container (Map Streamlit port 7860 to host)
docker run -p 7860:7860 --env-file backend/.env ai-code-assistant
```
Open `http://localhost:7860` in your web browser.

---

## 🚀 Deploying to Hugging Face Spaces

This repository is optimized for zero-config deployment to a Hugging Face Space using the **Docker SDK**.

1. Create a new **Space** on Hugging Face.
2. Select **Docker** as the SDK and choose **Blank** template.
3. Add your environment variables (like `GROQ_API_KEY` or `GEMINI_API_KEY`) under **Settings** ➔ **Variables and Secrets**.
4. Push your codebase to the Hugging Face Space repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push -f hf main
   ```
Hugging Face will automatically build and serve the Streamlit app on the default port `7860` using our custom `Dockerfile`!
