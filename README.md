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
- Persistent SQLite incident history with P01–P12 and severity filters, pattern/severity summaries, detail lookup, and CSV export
- Explicit incident-to-query trace linking with linked latency and answer/retrieval-mode breakdowns
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
| GET | `/` | API information |
| GET | `/health` | Database readiness check |
| GET | `/api/documents` | List indexed documents |
| POST | `/api/documents/upload` | Upload PDF, TXT, or Markdown |
| POST | `/api/query` | Retrieve evidence, diagnose, and optionally generate a Gemini answer |
| POST | `/api/evaluate` | Baseline retrieval evaluation |
| GET | `/api/traces` | List recent query traces |
| GET | `/api/traces/{trace_id}` | Retrieve one trace |
| GET | `/api/diagnostics/patterns` | List P01–P12 failure taxonomy |
| POST | `/api/diagnostics/incident` | Diagnose and persist a reported incident |
| GET | `/api/diagnostics/incidents` | Filter incident history and summaries |
| GET | `/api/diagnostics/incidents/{incident_id}` | Retrieve an incident report |
| GET | `/api/diagnostics/incidents.csv` | Export incident history |
| GET | `/api/diagnostics/incident-observability` | Summarize linked incident/trace observability |
| GET | `/api/benchmark/corpus` | Export indexed chunks for manual gold labeling |
| POST | `/api/benchmark/evaluate` | Evaluate one labeled retrieval case |
| POST | `/api/benchmark/compare` | Compare TF-IDF, hybrid, and hybrid+MMR for one query |
| POST | `/api/benchmark/run` | Run a labeled dataset across selected strategies |
| POST | `/api/benchmark/tune-mmr` | Sweep MMR diversity values on the same labeled dataset |
| POST | `/api/benchmark/regression` | Gate quality drops between comparable saved runs |
| GET | `/api/benchmark/snapshots/{snapshot_id}` | Read and integrity-check a benchmark snapshot |
| POST | `/api/benchmark/snapshots/{snapshot_id}/replay` | Replay saved corpus and dataset inputs |
| GET | `/api/benchmark/report` | Historical metric trends |
| GET | `/api/benchmark/report.csv` | Download benchmark history |
| GET | `/api/benchmark/insights` | Compare latest comparable strategies and recommendations |
| GET | `/api/benchmark/failure-analytics` | Rule-based aggregate failure signals |
| POST | `/api/quality/evaluate` | Gemini-assisted answer-grounding evaluation |
| GET | `/api/experiments` | List saved experiments |

All `/api/*` endpoints require a bearer token when `API_AUTH_TOKEN` is configured.

Example query body:

    {
      "question": "What is the main topic of this document?",
      "top_k": 4,
      "generate_answer": true
    }

Set generate_answer to false to isolate retrieval from generation.

## Production configuration and safety

The API reads comma-separated trusted browser origins from `CORS_ORIGINS`. The default permits only the local Vite origins. Wildcard origins are rejected because credentialed CORS must use explicit trusted origins. Set `LOG_LEVEL` to control structured request logging; logs include request ID, method, path, status, and duration, but intentionally omit request bodies and API keys.

The API returns an `X-Request-ID` response header and basic browser security headers. When `API_AUTH_TOKEN` is configured, every `/api/` route requires `Authorization: Bearer <token>`; the dashboard provides a session-only token field. `/health` remains public for container readiness. API rate limiting is process-local and keyed by the direct client IP. `GET /health` now checks database connectivity and returns HTTP 503 if SQLite is unavailable. Upload reads are capped at 10 MB plus one byte before processing, and Windows-style path separators in uploaded filenames are normalized before storing the basename.

For Docker, configure `CORS_ORIGINS`, `LOG_LEVEL`, `API_AUTH_TOKEN`, and `VITE_API_URL` in the environment before building. The backend image runs as a non-root user; both services expose container health checks, and the frontend waits for backend readiness. The browser must be able to reach the URL embedded in `VITE_API_URL`; never use the internal Compose hostname for a browser-facing build. `VITE_API_URL` is embedded in the frontend at build time; use the public browser-reachable API URL, not the internal Compose service hostname.

These are baseline hardening measures, not a complete production security review. Bearer-token API authentication can be enabled with API_AUTH_TOKEN; an in-memory per-process, per-client fixed-window rate limit is configurable with API_RATE_LIMIT and API_RATE_WINDOW_SECONDS. Set a strong random token before exposing the API. The limiter is not distributed across workers/replicas; use a gateway or shared store for multi-instance production. TLS termination, managed secret storage, persistent abuse monitoring, and a full security review are still required.

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
- [x] Persistent incident history, pattern/severity filters, summaries, and CSV export
- [x] Incident-to-query trace linking and descriptive observability correlation
- [x] Dense semantic embeddings and hybrid retrieval
- [x] MMR reranking and retrieval strategy comparison
- [x] Model-assisted faithfulness evaluation and labeled retrieval benchmark tooling
- [x] Persistent experiment runs and comparison dashboard
- [x] Batch dataset benchmarking with latency summaries
- [x] Explainable per-query retrieval failure diagnostics
- [x] Benchmark dataset validation against current corpus chunk IDs
- [x] Bounded MMR diversity tuning sweep with saved experiments
- [x] Benchmark regression gates for comparable saved runs
- [x] Historical benchmark reports with CSV export
- [x] Immutable benchmark snapshots and SHA-256 corpus/dataset fingerprints
- [x] Snapshot replay independent of the live indexed corpus
- [x] Explainable per-query retrieval failure diagnostics
- [x] Benchmark dataset validation against current corpus chunk IDs
- [x] MMR diversity tuning sweep with saved experiments
- [ ] PostgreSQL + pgvector storage option (not implemented)
- [x] Docker and CI workflow configuration (execution still pending verification)
- [x] Optional bearer-token API authentication and configurable process-local rate limiting
- [x] Production-oriented Docker images, health checks, non-root backend, and deployment checklist\n- [ ] Cloud deployment and public demo

## Project status and completion gates

**Application feature implementation:** the repository includes document ingestion, grounded RAG, hybrid retrieval, incident diagnostics, traces, labeled evaluation, immutable benchmark snapshots, replay, strategy comparison, failure analytics, MMR tuning, regression gates, and a React dashboard.

**Not yet verified / not included in a 100% completion claim:**
- GitHub Actions runs must finish successfully. A queued run is not a passing test.
- Docker images and the full Windows setup must be built and exercised.
- Render backend and frontend must both reach Live; health, authenticated API, browser CORS, upload, Gemini query, benchmark and incident workflows must be smoke-tested.
- The current Render Blueprint stores SQLite under `/tmp`, which is ephemeral. Documents, traces, incidents and benchmark history can be lost on restart/redeploy. Durable hosted use needs persistent storage and backups.
- PostgreSQL + pgvector is not implemented. Current dense retrieval uses Gemini embeddings stored with SQLite.
- A real manually labeled corpus benchmark and real Gemini answer-grounding review are required before publishing performance claims.

This repository is a portfolio/demo implementation, not a production-certified service. Do not publish API tokens or Gemini keys. See [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) for the remaining release checks.


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


## Deployment verification

After Render marks both services Live, follow [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md). A standard-library smoke test is available at `scripts/smoke_test.py`; it checks public health endpoints and confirms that protected API endpoints require the configured bearer token. The smoke test has not been run against the live deployment yet.
