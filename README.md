# RAG Failure Diagnostics & Evaluation Platform

A full-stack RAG workbench focused on diagnosing retrieval and answer-generation failures. Current capabilities include document ingestion, TF-IDF retrieval, transparent retrieval diagnostics, Gemini-grounded answer generation, and an initial automated test suite.

## Current capabilities

- Upload PDF, TXT, and Markdown files (10 MB limit)
- Extract text, split it into overlapping chunks, and persist chunks in SQLite
- Retrieve evidence using hybrid TF-IDF + Gemini semantic embeddings, with a TF-IDF-only fallback when embeddings are unavailable
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
- Retrieval: scikit-learn TF-IDF + Gemini Embedding (`gemini-embedding-001`) cosine similarity\n- Hybrid rank: weighted lexical/semantic score (45% / 55%) when dense vectors are available; transparent TF-IDF fallback otherwise
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

The regression suite covers SQLite persistence, chunking and retrieval, benchmark metrics, MMR strategies, answer-quality parsing, trace persistence, CORS and database readiness, plus API-level document upload and query workflows. API tests mock external Gemini calls so the suite does not require a live API key or incur model usage.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | API information |
| GET | /health | Health check |
| GET | /api/documents | List indexed documents |
| POST | /api/documents/upload | Upload and index a document |
| POST | /api/query | Retrieve, diagnose, and optionally generate a Gemini answer |
| POST | /api/evaluate | Compare retrieved context with a reference answer |\n| GET | /api/diagnostics/patterns | List supported P01–P12 failure patterns |\n| POST | /api/diagnostics/incident | Diagnose a reported RAG/LLM incident with Gemini |
| GET | /api/diagnostics/patterns | List supported P01–P12 failure patterns |
| POST | /api/diagnostics/incident | Diagnose a reported RAG/LLM incident with Gemini |

Example query body:

    {
      "question": "What is the main topic of this document?",
      "top_k": 4,
      "generate_answer": true
    }

Set generate_answer to false to isolate retrieval from generation.

## Production configuration and safety

The API reads comma-separated trusted browser origins from `CORS_ORIGINS`. The default permits only the local Vite origins. Wildcard origins are rejected because credentialed CORS must use explicit trusted origins. Set `LOG_LEVEL` to control structured request logging; logs include request ID, method, path, status, and duration, but intentionally omit request bodies and API keys.

The API returns an `X-Request-ID` response header and basic browser security headers. `GET /health` now checks database connectivity and returns HTTP 503 if SQLite is unavailable. Upload reads are capped at 10 MB plus one byte before processing, and Windows-style path separators in uploaded filenames are normalized before storing the basename.

For Docker, configure `CORS_ORIGINS`, `LOG_LEVEL`, and `VITE_API_URL` in the environment before building. `VITE_API_URL` is embedded in the frontend at build time; use the public browser-reachable API URL, not the internal Compose service hostname.

These are baseline hardening measures, not a complete production security review. Authentication, authorization, rate limiting, secret management, TLS termination, and external monitoring still need to be configured before exposing the service publicly.

## Query observability

Every `POST /api/query` response now includes a UUID `trace_id`, stage timings (`embedding_ms`, `retrieval_ms`, `generation_ms`, and `total_ms`), and a nullable `token_usage` field. Query traces are persisted in SQLite, including the retrieval/answer modes, top-k, retrieved chunk count, timestamp, and error details when a query fails.

Use `GET /api/traces?limit=30` for recent traces and `GET /api/traces/{trace_id}` for one trace. The frontend's Observability & Tracing panel displays recent timings and trace details.

Token counts and monetary cost are currently unavailable because the generation path does not yet expose provider usage metadata to the trace layer. The UI deliberately reports this as unavailable instead of estimating or inventing costs. Stage timings are server-side wall-clock measurements, not provider billing metrics.

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
- [x] Dense semantic embeddings and hybrid retrieval
- [x] MMR reranking and retrieval strategy comparison
- [x] Model-assisted faithfulness evaluation and labeled retrieval benchmark tooling
- [x] Persistent experiment runs and comparison dashboard
- [x] Batch dataset benchmarking with latency summaries
- [x] Benchmark regression gates for comparable saved runs
- [x] Historical benchmark reports with CSV export
- [x] Batch dataset benchmarking with latency summaries
- [x] Benchmark regression gates for comparable saved runs
- [ ] PostgreSQL + pgvector storage option
- [x] Docker and CI workflow configuration (execution still pending verification)\n- [ ] Deployment and public demo

## Project status

**Active development — retrieval evaluation, answer grounding, experiment tracking, and query observability implemented.** Repository changes have not yet been run in your Windows environment or against a live Gemini API key. Test results, live API behavior, and deployment will be reported only after verification.


## Docker

1. Create `backend/.env` from `backend/.env.example` and set your Gemini key.
2. From the repository root run:

    docker compose up --build

3. Open the frontend at http://localhost:5173 and API docs at http://localhost:8000/docs.

The SQLite database is stored in a named Docker volume. Do not commit `backend/.env`.

## Retrieval behavior

Newly uploaded chunks are embedded using Gemini Embedding. Query text is embedded with the retrieval-query task type. When both query and stored vectors are available, the ranker combines lexical TF-IDF similarity (45%) and semantic cosine similarity (55%). If the embedding key is absent, quota is exhausted, or the embedding endpoint fails, retrieval explicitly falls back to TF-IDF. Existing documents uploaded before embedding support will continue to work through this fallback; re-upload them to generate vectors.

The current ranking is a baseline hybrid ranker, not a learned reranker. Scores are useful for ranking and diagnostics, not calibrated probabilities.

## Current limitations

- Embedding generation currently makes one API call per chunk; very large uploads may hit free-tier rate limits.
- The incident diagnosis is LLM-assisted hypothesis generation, not verified root-cause analysis.
- Reference-answer coverage remains lexical; a curated benchmark and faithfulness evaluation are still required.
- Docker and CI configuration are committed but must pass local/remote builds before being described as verified.
