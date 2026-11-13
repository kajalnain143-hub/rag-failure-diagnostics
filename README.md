# RAG Failure Diagnostics & Evaluation Platform

A lightweight, full-stack MVP for inspecting retrieval quality and diagnosing common failure patterns in Retrieval-Augmented Generation (RAG) workflows.

> **Current implementation:** document ingestion, text chunking, TF-IDF retrieval, retrieval diagnostics, a baseline lexical evaluation metric, and a React dashboard. This version does not yet generate LLM answers or claim to detect hallucinations.

## Features

- Upload PDF, TXT, and Markdown documents
- Extract text and split it into overlapping chunks
- Store documents and chunks in SQLite
- Retrieve relevant chunks using TF-IDF and cosine similarity
- Inspect similarity scores and retrieval diagnostics
- Compare retrieved context with a supplied reference answer
- Use a React dashboard to upload, query, and inspect results

## Architecture

```text
rag-failure-diagnostics/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── diagnostics.py
│   │   ├── main.py
│   │   ├── rag_engine.py
│   │   └── schemas.py
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── main.jsx
    │   └── styles.css
    ├── index.html
    ├── package.json
    └── vite.config.js
```

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- npm

## Run locally on Windows

### 1. Clone the repository

```bash
git clone https://github.com/kajalnain143-hub/rag-failure-diagnostics.git
cd rag-failure-diagnostics
```

### 2. Start the backend

Open a terminal:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000. Interactive API documentation: http://localhost:8000/docs.

### 3. Start the frontend

Open a second terminal from the project root:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, normally http://localhost:5173.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| GET | `/api/documents` | List indexed documents |
| POST | `/api/documents/upload` | Upload and index a document |
| POST | `/api/query` | Retrieve chunks and diagnose retrieval |
| POST | `/api/evaluate` | Compare retrieved context with a reference answer |

## Evaluation notes

The current retrieval engine uses TF-IDF lexical features and cosine similarity. The reference-answer coverage metric measures lexical overlap only. These metrics are useful for a transparent baseline, but they are not substitutes for semantic evaluation, factuality checks, or human review.

## Roadmap

- [x] Initial FastAPI service
- [x] Document upload and text extraction
- [x] Chunking and SQLite persistence
- [x] TF-IDF retrieval baseline
- [x] Retrieval diagnostics and lexical evaluation
- [x] React dashboard
- [ ] Automated test suite
- [ ] Embedding-based semantic retrieval
- [ ] LLM answer generation with source citations
- [ ] RAGAS-based evaluation
- [ ] Experiment history and comparison
- [ ] Docker deployment

## Project status

**MVP under active development.** Features will be expanded and validated as the project evolves.
