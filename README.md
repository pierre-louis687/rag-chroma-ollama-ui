rag-chroma-ollama-ui

Minimal local RAG implementation using Chroma (vector database) and Ollama (LLM + embeddings) with a lightweight Flask web UI.
Designed for CPU environments and advanced developers.

Overview

This project implements a fully local Retrieval-Augmented Generation (RAG) pipeline:
PDF documents are indexed into a Chroma vector store.
Embeddings are generated via Ollama (nomic-embed-text).
User queries are embedded and matched against stored vectors.
Relevant chunks are injected into a prompt.
A local Ollama LLM generates the final answer.
A Flask-based UI exposes the system via HTTP.

No external APIs required.
Everything runs locally.

Architecture
User (Web UI)
        │
        ▼
     Flask (app.py)
        │
        ▼
   query_core.py
        │
        ├── Embedding model (Ollama: nomic-embed-text)
        │
        ├── Vector search (Chroma)
        │
        └── LLM generation (Ollama: qwen2.5 / mistral / etc.)

Core Components

Chroma → persistent vector database (chroma/)
Ollama → local embedding + LLM inference
Flask → minimal web interface
populate_database2.py → indexing pipeline
query_core.py → RAG orchestration layer

Features

Local PDF indexing
Configurable k retrieval
Adjustable chunk size
CPU-friendly model support
Source filtering
CLI querying support
No cloud dependencies

Requirements

Python 3.10+
Ollama installed and running
Apache2 (if used behind reverse proxy)
CPU Optimization (Recommended)

For better performance on multi-core CPUs:

sudo systemctl edit ollama
Add:
[Service]
Environment="OLLAMA_NUM_THREADS=4"

Then restart:
sudo systemctl daemon-reload
sudo systemctl restart ollama

Adjust thread count according to your CPU.

Installation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Create the directory data for PDF files to import


Model Setup

Pull required Ollama models:
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
You may substitute other LLMs depending on your hardware.

Indexing Documents
Place your PDF files into the data/ directory.

Then build the Chroma database:
python populate_database2.py --reset
The --reset option clears the existing vector store.

Running the Web UI
python app.py

Open:
http://127.0.0.1:8085

CLI Usage (No Flask)
You can query directly from the terminal:
python query_data2.py "how to code in HTML" --source "data/doc_html.pdf" --k 8 --debug

Project Structure
rag-chroma-ollama-ui/
│
├── app.py
├── query_core.py
├── populate_database2.py
├── get_embedding_function.py
├── query_data2.py
│
├── chroma/          # Vector database (ignored in Git)
├── data/            # PDF input folder (PDFs ignored in Git)
│
├── templates/       # Flask HTML templates
├── requirements.txt
└── .env.example


Design Philosophy
Fully local
Transparent architecture
Explicit control over retrieval parameters
Developer-first approach
Minimal abstraction layers

Security Note
This project is intended for local use.
If exposing publicly:
Add authentication
Use a reverse proxy
Implement rate limiting
Secure Ollama endpoints

License MIT (recommended for open technical projects).

contact : contact@carburantpascher.org
