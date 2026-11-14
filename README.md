# RAG Failure Diagnostics & Evaluation Platform

A full-stack RAG workbench focused on diagnosing retrieval and answer-generation failures. Current capabilities include document ingestion, TF-IDF retrieval, transparent retrieval diagnostics, Gemini-grounded answer generation, and an initial automated test suite.

## Current capabilities

- Upload PDF, TXT, and Markdown files (10 MB limit)
- Extract text, split it into overlapping chunks, and persist chunks in SQLite
- Retrieve evidence using TF-IDF and cosine similarity
- Diagnose no-context, weak-retrieval, and retrieval-candidate states
- Generate grounded answers with Gemini 2.5 Flash
- Instruct Gemini to cite source labels such as [S1] and abstain when evidence is insufficient
- Return source metadata, retrieval scores, and answer-generation status
- Disable generation to inspect retrieval independently
- Baseline lexical reference-answer coverage metric
- Gemini-powered incident triage based on a 12-pattern RAG/LLM failure taxonomy (P01–P12)
- Structured diagnosis output: primary/secondary pattern, severity, confidence, reasoning, missing evidence, minimal fix, and verification steps
- Strict validation of model-generated JSON before returning a diagnosis
- Third-party attribution and Apache 2.0 license notice for the adapted clinic taxonomy
- Pytest tests for chunking, retrieval, diagnostics, and evaluation

> Source labels improve traceability but do not guarantee factual correctness. The current retrieval and reference-answer metrics are lexical baselines, not semantic relevance or hallucination detection.

## Architecture

    Documents (PDF/TXT/MD) -> extraction -> chunks -> SQLite
    Question -> TF-IDF retrieval -> ranked evidence -> diagnostics
                                                -> Gemini grounded generation
                                                -> API response with source metadata
    React dashboard <---------------------------+

## Stack

- Backend: Python, FastAPI, Pydantic
- LLM: Google Gemini Developer API via google-genai
- Retrieval baseline: scikit-learn TF-IDF + cosine similarity
- Storage: SQLite
- Frontend: React + Vite
- Tests: pytest

## Setup (Windows 11)

### 1. Clone and prepare backend

    git clone https://github.com/kajalnain143-hub/rag-failure-diagnostics.git
    cd rag-failure-diagnostics/backend
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

### 2. Configure Gemini

Copy backend/.env.example to backend/.env and add your Google AI Studio key:

    GEMINI_API_KEY=your_key_here
    GEMINI_MODEL=gemini-2.5-flash

Generate a key at https://aistudio.google.com/app/apikey. Never commit .env or paste the key into source code. Gemini free-tier model availability and request limits can vary by model, region, and account. Check https://ai.google.dev/gemini-api/docs/pricing before use.

### 3. Start backend

From the backend directory:

    uvicorn app.main:app --reload

API docs: http://localhost:8000/docs

### 4. Start frontend

Open another terminal from the repository root:

    cd frontend
    npm install
    npm run dev

Open the Vite URL, normally http://localhost:5173.

## Run tests

From backend with the virtual environment active:

    pytest

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | API information |
| GET | /health | Health check |
| GET | /api/documents | List indexed documents |
| POST | /api/documents/upload | Upload and index a document |
| POST | /api/query | Retrieve, diagnose, and optionally generate a Gemini answer |
| POST | /api/evaluate | Compare retrieved context with a reference answer |

Example query body:

    {
      "question": "What is the main topic of this document?",
      "top_k": 4,
      "generate_answer": true
    }

Set generate_answer to false to isolate retrieval from generation.

## Evaluation honesty

The expected-answer coverage score measures lexical term overlap between the reference answer and retrieved context. It is not a semantic relevance score, faithfulness score, or hallucination detector. Gemini is instructed to ground answers in retrieved chunks and cite them, but generated citations still require evaluation.

## Roadmap

- [x] FastAPI service and document ingestion
- [x] SQLite persistence and TF-IDF retrieval baseline
- [x] Initial retrieval diagnostics and lexical evaluation
- [x] Gemini grounded answer generation and source metadata
- [x] Initial automated unit tests
- [x] Gemini-powered incident diagnostics clinic (adapted from Darpan Sharma's Apache-2.0-licensed reference; see THIRD_PARTY_NOTICES.md)
- [x] Structured diagnosis validation and incident endpoint
- [ ] Dense semantic embeddings and hybrid retrieval
- [ ] Reranking and retrieval ablation experiments
- [ ] Faithfulness and answer-relevance evaluation with curated benchmark
- [ ] Persistent experiment runs and comparison dashboard
- [ ] PostgreSQL + pgvector storage option
- [ ] Docker, CI workflow, deployment, and demo

## Project status

**Active development — Phase 1 committed.** Repository changes have not yet been run in your Windows environment or against a live Gemini API key. Test results, live API behavior, and deployment will be reported only after verification.
