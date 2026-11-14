from io import BytesIO, StringIO
import csv
from pathlib import Path
import json
import logging
import os
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pypdf import PdfReader

from .database import get_all_chunks, get_documents, get_connection, init_db, save_document, create_database_backup
from .diagnostics import diagnose, evaluate_retrieval
from .llm import LLMGenerationError, LLMNotConfiguredError, generate_grounded_answer, verify_live_api
from .embeddings import EMBEDDING_MODEL, embed_query, embed_text
from .failure_clinic import PATTERNS, diagnose_incident
from .incident_schemas import IncidentDiagnosisRequest
from .rag_engine import retrieve, split_into_chunks
from .schemas import EvaluationRequest, QueryRequest
from .experiment_store import init_experiment_store, list_experiments, save_experiment
from .evaluation_metrics import aggregate_metrics, evaluate_case
from .answer_quality import context_precision_recall, evaluate_answer_grounding
from .trace_store import init_trace_store, save_trace, list_traces, get_trace
from .benchmark_runner import ALLOWED_STRATEGIES, run_dataset
from .benchmark_optimizer import rank_mmr_sweep
from .benchmark_regression import compare_benchmark_runs
from .benchmark_reports import build_benchmark_report, build_benchmark_insights, REPORT_METRICS
from .benchmark_failure_analytics import analyze_benchmark_failures
from .query_failure_diagnostics import diagnose_query_failure
from .incident_store import init_incident_store, save_incident, list_incidents, get_incident, incident_pattern_summary, incident_trace_analytics
from .benchmark_snapshots import init_snapshot_store, create_snapshot, get_snapshot, SnapshotIntegrityError
from .security import FixedWindowRateLimiter, valid_bearer_token

app = FastAPI(
    title="RAG Failure Diagnostics API",
    description="A RAG workbench for grounded answers, retrieval diagnostics, and evaluation.",
    version="0.2.0",
)
def configured_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    origins = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    if not origins or "*" in origins:
        raise RuntimeError("CORS_ORIGINS must contain explicit trusted origins; wildcard origins are not allowed.")
    return origins


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(message)s")
logger = logging.getLogger("rag_api")
GEMINI_LIVE_STATUS = {"status": "not_checked"}
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "").strip()
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "120"))
API_RATE_WINDOW_SECONDS = int(os.getenv("API_RATE_WINDOW_SECONDS", "60"))
api_rate_limiter = FixedWindowRateLimiter(API_RATE_LIMIT, API_RATE_WINDOW_SECONDS)
if not API_AUTH_TOKEN:
    logger.warning("API_AUTH_TOKEN is unset; API bearer authentication is disabled.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID", "Authorization"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = str(uuid4())
    started = perf_counter()
    if request.url.path.startswith("/api/") and request.method != "OPTIONS":
        if not valid_bearer_token(request.headers.get("Authorization"), API_AUTH_TOKEN):
            return JSONResponse(status_code=401, content={"detail": "A valid Bearer API token is required."},
                                headers={"WWW-Authenticate": "Bearer", "X-Request-ID": request_id})
        client_key = request.client.host if request.client else "unknown"
        allowed, retry_after = api_rate_limiter.allow(client_key)
        if not allowed:
            return JSONResponse(status_code=429, content={"detail": "API rate limit exceeded. Retry later."},
                                headers={"Retry-After": str(retry_after), "X-Request-ID": request_id})
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        logger.exception(json.dumps({
            "event": "http_request_failed", "request_id": request_id,
            "method": request.method, "path": request.url.path, "duration_ms": elapsed_ms,
        }))
        raise
    elapsed_ms = round((perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Frame-Options"] = "DENY"
    logger.info(json.dumps({
        "event": "http_request", "request_id": request_id,
        "method": request.method, "path": request.url.path,
        "status_code": response.status_code, "duration_ms": elapsed_ms,
    }))
    return response


@app.on_event("startup")
def startup():
    global GEMINI_LIVE_STATUS
    init_db()
    init_experiment_store()
    init_trace_store()
    init_snapshot_store()
    init_incident_store()
    GEMINI_LIVE_STATUS = verify_live_api()
    logger.info(json.dumps({"event": "gemini_live_verification", **GEMINI_LIVE_STATUS}))


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Text files must use UTF-8 encoding.") from exc
    if suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise HTTPException(status_code=400, detail="Password-protected PDFs are not supported.")
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail="The PDF could not be parsed.") from exc
    raise HTTPException(status_code=400, detail="Only PDF, TXT, and Markdown files are supported.")


@app.get("/")
def read_root():
    return {"message": "RAG Failure Diagnostics API is running.", "version": app.version, "docs": "/docs"}


@app.get("/health/gemini")
def gemini_health_check():
    """Return the last startup live Gemini generation/embedding verification; never expose credentials."""
    return GEMINI_LIVE_STATUS


@app.get("/health")
def health_check():
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        logger.exception("Database readiness check failed")
        raise HTTPException(status_code=503, detail="Database is unavailable.") from exc
    return {"status": "healthy", "database": "ready", "service": "rag-failure-diagnostics-api"}


@app.get("/api/admin/database-backup")
def download_database_backup():
    """Download a consistent SQLite backup; protect this endpoint with API_AUTH_TOKEN."""
    try:
        backup = create_database_backup()
    except Exception as exc:
        logger.exception("Database backup failed")
        raise HTTPException(status_code=503, detail="Database backup could not be created.") from exc
    return Response(
        content=backup,
        media_type="application/vnd.sqlite3",
        headers={"Content-Disposition": 'attachment; filename="rag-diagnostics-backup.sqlite3"',
                 "Cache-Control": "no-store"},
    )


@app.get("/api/documents")
def list_documents():
    return {"documents": get_documents()}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path((file.filename or "uploaded-document").replace("\\", "/")).name
    content = await file.read(10 * 1024 * 1024 + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum upload size is 10 MB.")
    text = extract_text(filename, content)
    chunks = split_into_chunks(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text was found in the document.")
    embeddings = [embed_text(chunk) for chunk in chunks]
    document_id = save_document(filename, chunks, embeddings, EMBEDDING_MODEL)
    return {"message": "Document uploaded and indexed successfully.", "document_id": document_id,
            "filename": filename, "chunk_count": len(chunks)}


@app.post("/api/query")
def query_documents(request: QueryRequest):
    trace_id = str(uuid4())
    started = perf_counter()
    timings = {"embedding_ms": 0.0, "retrieval_ms": 0.0, "generation_ms": 0.0}
    answer_mode, retrieval_mode, token_usage = "error", "none", None
    try:
        chunks = get_all_chunks()
        if not chunks:
            raise HTTPException(status_code=400, detail="Upload a document before asking a question.")
        stage = perf_counter()
        query_vector = embed_query(request.question)
        timings["embedding_ms"] = round((perf_counter() - stage) * 1000, 2)
        stage = perf_counter()
        results = retrieve(request.question, chunks, request.top_k, query_embedding=query_vector)
        timings["retrieval_ms"] = round((perf_counter() - stage) * 1000, 2)
        retrieval_mode = results[0].get("retrieval_mode") if results else "none"
        diagnostic = diagnose(request.question, results)
        sources = [
            {"label": f"S{index}", "chunk_id": item["chunk_id"], "document_id": item["document_id"],
             "filename": item["filename"], "score": item["score"], "content": item["content"]}
            for index, item in enumerate(results, start=1)
        ]
        if not results or results[0]["score"] < 0.05:
            answer, answer_mode = "I could not find sufficiently relevant context in the indexed documents.", "insufficient_context"
        elif not request.generate_answer:
            answer, answer_mode = "Answer generation was disabled. Review the retrieved evidence below.", "retrieval_only"
        else:
            stage = perf_counter()
            try:
                answer = generate_grounded_answer(request.question, results)
                answer_mode = "gemini"
            except LLMNotConfiguredError:
                answer, answer_mode = "Gemini API key is not configured. Add GEMINI_API_KEY to backend/.env to enable generation.", "llm_not_configured"
            except LLMGenerationError as exc:
                answer, answer_mode = str(exc), "llm_error"
            finally:
                timings["generation_ms"] = round((perf_counter() - stage) * 1000, 2)
        trace = {"trace_id": trace_id, "question": request.question, "answer_mode": answer_mode,
                 "retrieval_mode": retrieval_mode, "top_k": request.top_k,
                 "timings": {**timings, "total_ms": round((perf_counter() - started) * 1000, 2)},
                 "token_usage": token_usage, "retrieved_count": len(results), "error": None}
        save_trace(trace)
        return {"trace_id": trace_id, "timings": trace["timings"], "token_usage": token_usage,
                "question": request.question, "answer": answer, "answer_mode": answer_mode,
                "model": "gemini-2.5-flash" if answer_mode == "gemini" else None,
                "retrieved_chunks": results, "sources": sources, "diagnostics": diagnostic,
                "retrieval_mode": retrieval_mode}
    except Exception as exc:
        message = str(exc.detail if isinstance(exc, HTTPException) else exc)[:1000]
        save_trace({"trace_id": trace_id, "question": request.question, "answer_mode": answer_mode,
                    "retrieval_mode": retrieval_mode, "top_k": request.top_k,
                    "timings": {**timings, "total_ms": round((perf_counter() - started) * 1000, 2)},
                    "token_usage": token_usage, "retrieved_count": 0, "error": message})
        raise


@app.get("/api/traces")
def get_recent_traces(limit: int = 30):
    return {"traces": list_traces(limit)}


@app.get("/api/traces/{trace_id}")
def get_trace_detail(trace_id: str):
    trace = get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return {"trace": trace}


@app.post("/api/evaluate")
def evaluate(request: EvaluationRequest):
    chunks = get_all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="Upload a document before running evaluation.")
    results = retrieve(request.question, chunks, request.top_k)
    return {"evaluation": evaluate_retrieval(request.question, request.expected_answer, results),
            "diagnostics": diagnose(request.question, results), "retrieved_chunks": results,
            "retrieval_mode": results[0].get("retrieval_mode") if results else "none"}



@app.get("/api/diagnostics/patterns")
def list_failure_patterns():
    """Return the supported incident taxonomy for the diagnostics UI."""
    return {"taxonomy_version": "1.0", "count": len(PATTERNS), "patterns": PATTERNS}


@app.post("/api/diagnostics/incident")
def diagnose_rag_incident(request: IncidentDiagnosisRequest):
    """Classify a reported RAG/LLM incident and propose verifiable structural fixes."""
    try:
        diagnosis = diagnose_incident(
            request.description,
            logs=request.logs,
            expected_behavior=request.expected_behavior,
        )
        incident = {
            "description": request.description,
            "expected_behavior": request.expected_behavior,
            "logs": request.logs,
            "trace_id": request.trace_id,
        }
        if request.trace_id and get_trace(request.trace_id) is None:
            raise HTTPException(status_code=404, detail="Linked query trace was not found.")
        saved = save_incident(incident, diagnosis)
        return {"incident": incident, "diagnosis": diagnosis, "incident_id": saved["incident_id"],
                "created_at": saved["created_at"]}
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail="Gemini API key is not configured.") from exc
    except LLMGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/diagnostics/incidents")
def get_incident_history(limit: int = 50, pattern_id: str | None = None, severity: str | None = None):
    try:
        return {"incidents": list_incidents(limit, pattern_id, severity),
                "summary": incident_pattern_summary()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/diagnostics/incidents/{incident_id}")
def get_incident_report(incident_id: str):
    incident = get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident report not found.")
    return {"incident": incident}


@app.get("/api/diagnostics/incident-observability")
def get_incident_observability():
    return incident_trace_analytics()


@app.get("/api/diagnostics/incidents.csv")
def export_incident_history(pattern_id: str | None = None, severity: str | None = None):
    try:
        incidents = list_incidents(200, pattern_id, severity)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["incident_id", "created_at", "primary_pattern_id", "secondary_pattern_ids",
                     "severity", "confidence", "description", "expected_behavior", "logs"])
    for item in incidents:
        writer.writerow([item["incident_id"], item["created_at"], item["primary_pattern_id"],
                         ";".join(item["secondary_pattern_ids"]), item["severity"], item["confidence"],
                         item["description"], item["expected_behavior"], item["logs"]])
    return Response(content=output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="incident-history.csv"'})


@app.get("/api/experiments")
def get_experiment_history(limit: int = 20):
    """Return recent saved retrieval/evaluation experiment runs."""
    return {"experiments": list_experiments(limit)}


@app.post("/api/experiments")
def create_experiment(payload: dict):
    """Persist a named experiment configuration and its measured metrics."""
    name = str(payload.get("name", "")).strip()
    config = payload.get("config", {})
    metrics = payload.get("metrics", {})
    notes = str(payload.get("notes", ""))
    if not name or not isinstance(config, dict) or not isinstance(metrics, dict):
        raise HTTPException(status_code=400, detail="Provide a name plus object-valued config and metrics.")
    try:
        run_id = save_experiment(name, config, metrics, notes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": run_id, "message": "Experiment saved."}


@app.post("/api/benchmark/evaluate")
def evaluate_benchmark_case(payload: dict):
    """Evaluate retrieved chunks against a manually labeled benchmark case."""
    required = ("query_id", "question", "relevant_chunk_ids", "reference_answer")
    if any(key not in payload for key in required):
        raise HTTPException(status_code=400, detail=f"Required fields: {', '.join(required)}")
    if not isinstance(payload["relevant_chunk_ids"], list):
        raise HTTPException(status_code=400, detail="relevant_chunk_ids must be a list.")
    chunks = get_all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="Upload a document before running a benchmark.")
    question = str(payload["question"])
    results = retrieve(question, chunks, int(payload.get("k", 5)))
    metrics = evaluate_case(payload, results)
    return {"metrics": metrics, "retrieved_chunks": results}



@app.get("/api/benchmark/corpus")
def export_benchmark_corpus():
    """Expose real indexed chunk IDs and text to author correctly labeled datasets."""
    chunks = get_all_chunks()
    return {
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": item["id"],
                "document_id": item["document_id"],
                "filename": item["filename"],
                "chunk_index": item.get("chunk_index"),
                "content": item["content"],
            }
            for item in chunks[:500]
        ],
        "dataset_case_template": {
            "query_id": "replace-with-stable-id",
            "question": "Write a question answerable from the indexed corpus.",
            "relevant_chunk_ids": ["replace-with-real-chunk-id"],
            "reference_answer": "Write a concise answer supported by the relevant chunk.",
            "k": 5,
        },
        "note": "Replace the placeholder relevant_chunk_ids with actual integer chunk_id values. Label relevance manually; do not treat every retrieved chunk as gold evidence.",
    }


@app.post("/api/benchmark/compare")
def compare_retrieval_strategies(payload: dict):
    """Run one labeled query through TF-IDF, hybrid, and hybrid+MMR."""
    required = ("query_id", "question", "relevant_chunk_ids", "reference_answer")
    if any(key not in payload for key in required):
        raise HTTPException(status_code=400, detail=f"Required fields: {', '.join(required)}")
    if not isinstance(payload["relevant_chunk_ids"], list):
        raise HTTPException(status_code=400, detail="relevant_chunk_ids must be a list.")
    chunks = get_all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="Upload a document before running a benchmark.")
    question = str(payload["question"]).strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty.")
    try:
        k = max(1, min(int(payload.get("k", 5)), 10))
        diversity = float(payload.get("mmr_diversity", 0.35))
        query_vector = embed_query(question)
        strategies = ("tfidf", "hybrid", "hybrid_mmr")
        comparisons = {}
        for strategy in strategies:
            retrieved = retrieve(
                question, chunks, k, query_embedding=query_vector,
                strategy=strategy, mmr_diversity=diversity,
            )
            case_metrics = evaluate_case(payload, retrieved, k)
            comparisons[strategy] = {
                "metrics": case_metrics,
                "failure_diagnostics": diagnose_query_failure(case_metrics),
                "retrieved_chunks": retrieved,
            }
        summary = aggregate_metrics([row["metrics"] for row in comparisons.values()])
        best_strategy = max(
            strategies,
            key=lambda name: (
                comparisons[name]["metrics"]["ndcg_at_k"],
                comparisons[name]["metrics"]["mrr"],
                comparisons[name]["metrics"]["recall_at_k"],
            ),
        )
        return {
            "query_id": payload["query_id"],
            "comparison": comparisons,
            "best_strategy": best_strategy,
            "selection_note": "Best strategy is ranked by nDCG@K, then MRR, then Recall@K for this query only.",
            "metric_note": summary.get("metric_note", "Metrics are computed against manually labeled chunk IDs."),
        }
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc



@app.post("/api/benchmark/run")
def run_batch_benchmark(payload: dict):
    """Run a bounded labeled dataset across one or more retrieval strategies."""
    dataset_name = str(payload.get("dataset_name", "unnamed-dataset")).strip()
    cases = payload.get("cases")
    strategies = payload.get("strategies", ["tfidf", "hybrid", "hybrid_mmr"])
    diversity = payload.get("mmr_diversity", 0.35)
    if not dataset_name or len(dataset_name) > 120:
        raise HTTPException(status_code=400, detail="dataset_name must contain 1–120 characters.")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        raise HTTPException(status_code=400, detail="cases must be an array containing 1–100 labeled cases.")
    if (not isinstance(strategies, list) or not strategies
            or any(not isinstance(item, str) for item in strategies)
            or len(set(strategies)) != len(strategies)):
        raise HTTPException(status_code=400, detail="strategies must be a non-empty array without duplicates.")
    if any(strategy not in ALLOWED_STRATEGIES for strategy in strategies):
        raise HTTPException(status_code=400, detail=f"strategies must be selected from: {', '.join(sorted(ALLOWED_STRATEGIES))}.")
    if isinstance(diversity, bool) or not isinstance(diversity, (int, float)) or not 0 <= diversity <= 1:
        raise HTTPException(status_code=400, detail="mmr_diversity must be between 0 and 1.")
    seen_ids = set()
    clean_cases = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise HTTPException(status_code=400, detail=f"Case {index + 1} must be an object.")
        query_id, question = case.get("query_id"), case.get("question")
        relevant = case.get("relevant_chunk_ids")
        reference = case.get("reference_answer")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in seen_ids:
            raise HTTPException(status_code=400, detail=f"Case {index + 1} needs a unique non-empty query_id.")
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise HTTPException(status_code=400, detail=f"Case {query_id} needs a question of 1–2000 characters.")
        if not isinstance(relevant, list) or not relevant or any(isinstance(item, bool) or not isinstance(item, int) for item in relevant):
            raise HTTPException(status_code=400, detail=f"Case {query_id} needs a non-empty integer relevant_chunk_ids array.")
        if not isinstance(reference, str) or not reference.strip():
            raise HTTPException(status_code=400, detail=f"Case {query_id} needs a non-empty reference_answer.")
        raw_k = case.get("k", 5)
        if isinstance(raw_k, bool) or not isinstance(raw_k, int):
            raise HTTPException(status_code=400, detail=f"Case {query_id}: k must be an integer from 1 to 10.")
        k = raw_k
        if not 1 <= k <= 10:
            raise HTTPException(status_code=400, detail=f"Case {query_id}: k must be an integer from 1 to 10.")
        seen_ids.add(query_id)
        clean_cases.append({"query_id": query_id, "question": question.strip(),
                            "relevant_chunk_ids": relevant, "reference_answer": reference, "k": k})
    chunks = get_all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="Upload documents before running a batch benchmark.")
    try:
        snapshot = create_snapshot(dataset_name, clean_cases, chunks, float(diversity))
        return run_dataset(dataset_name, clean_cases, strategies, chunks,
                           embed_query, retrieve, save_experiment, float(diversity), snapshot)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/benchmark/tune-mmr")
def tune_mmr_parameters(payload: dict):
    """Run a bounded MMR-diversity sweep on identical labeled corpus inputs."""
    dataset_name = str(payload.get("dataset_name", "mmr-tuning")).strip()
    cases = payload.get("cases")
    diversity_values = payload.get("diversity_values", [0.0, 0.2, 0.35, 0.5, 0.7, 1.0])
    if not dataset_name or len(dataset_name) > 120:
        raise HTTPException(status_code=400, detail="dataset_name must contain 1–120 characters.")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 50:
        raise HTTPException(status_code=400, detail="cases must contain 1–50 labeled cases for a tuning sweep.")
    if (not isinstance(diversity_values, list) or not 2 <= len(diversity_values) <= 7
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not 0 <= value <= 1 for value in diversity_values)
            or len(set(diversity_values)) != len(diversity_values)):
        raise HTTPException(status_code=400, detail="diversity_values must contain 2–7 unique numbers between 0 and 1.")
    seen_ids, clean_cases = set(), []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise HTTPException(status_code=400, detail=f"Case {index + 1} must be an object.")
        query_id, question = case.get("query_id"), case.get("question")
        relevant, reference = case.get("relevant_chunk_ids"), case.get("reference_answer")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in seen_ids:
            raise HTTPException(status_code=400, detail=f"Case {index + 1} needs a unique non-empty query_id.")
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise HTTPException(status_code=400, detail=f"Case {query_id} needs a question of 1–2000 characters.")
        if (not isinstance(relevant, list) or not relevant
                or any(isinstance(item, bool) or not isinstance(item, int) for item in relevant)
                or len(set(relevant)) != len(relevant)):
            raise HTTPException(status_code=400, detail=f"Case {query_id} needs unique integer relevant_chunk_ids.")
        if not isinstance(reference, str) or not reference.strip():
            raise HTTPException(status_code=400, detail=f"Case {query_id} needs a non-empty reference_answer.")
        k = case.get("k", 5)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
            raise HTTPException(status_code=400, detail=f"Case {query_id}: k must be an integer from 1 to 10.")
        seen_ids.add(query_id)
        clean_cases.append({"query_id": query_id, "question": question.strip(),
                            "relevant_chunk_ids": relevant, "reference_answer": reference, "k": k})
    chunks = get_all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="Upload documents before tuning MMR.")
    results = []
    try:
        for value in diversity_values:
            diversity = float(value)
            snapshot = create_snapshot(dataset_name, clean_cases, chunks, diversity)
            run = run_dataset(dataset_name, clean_cases, ["hybrid_mmr"], chunks,
                              embed_query, retrieve, save_experiment, diversity, snapshot)
            strategy_result = run["strategies"]["hybrid_mmr"]
            results.append({
                "mmr_diversity": diversity,
                "experiment_id": strategy_result["experiment_id"],
                "snapshot_id": snapshot.get("snapshot_id"),
                "metrics": strategy_result["metrics"],
            })
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "dataset_name": dataset_name,
        "case_count": len(clean_cases),
        "sweep_count": len(results),
        **rank_mmr_sweep(results),
    }


@app.post("/api/benchmark/regression")
def benchmark_regression_check(payload: dict):
    """Compare two saved batch benchmark runs and flag quality regressions."""
    baseline_id = payload.get("baseline_run_id")
    current_id = payload.get("current_run_id")
    tolerance = payload.get("max_metric_drop", 0.05)
    for field, value in (("baseline_run_id", baseline_id), ("current_run_id", current_id)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise HTTPException(status_code=400, detail=f"{field} must be a positive integer.")
    if baseline_id == current_id:
        raise HTTPException(status_code=400, detail="Choose two different experiment runs.")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 1:
        raise HTTPException(status_code=400, detail="max_metric_drop must be between 0 and 1.")
    runs = {run["id"]: run for run in list_experiments(100)}
    if baseline_id not in runs or current_id not in runs:
        raise HTTPException(status_code=404, detail="One or both runs were not found in the latest 100 saved experiments.")
    try:
        return compare_benchmark_runs(runs[baseline_id], runs[current_id], float(tolerance))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/benchmark/snapshots/{snapshot_id}")
def read_benchmark_snapshot(snapshot_id: int):
    try:
        snapshot = get_snapshot(snapshot_id)
    except SnapshotIntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Benchmark snapshot not found.")
    return snapshot


@app.post("/api/benchmark/snapshots/{snapshot_id}/replay")
def replay_benchmark_snapshot(snapshot_id: int, payload: dict):
    try:
        snapshot = get_snapshot(snapshot_id)
    except SnapshotIntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Benchmark snapshot not found.")
    strategies = payload.get("strategies", ["tfidf", "hybrid", "hybrid_mmr"])
    if (not isinstance(strategies, list) or not strategies
            or any(not isinstance(item, str) for item in strategies)
            or len(set(strategies)) != len(strategies)
            or any(item not in ALLOWED_STRATEGIES for item in strategies)):
        raise HTTPException(status_code=400, detail=f"strategies must be unique values from: {', '.join(sorted(ALLOWED_STRATEGIES))}.")
    try:
        return run_dataset(snapshot["dataset_name"], snapshot["cases"], strategies,
                           snapshot["chunks"], embed_query, retrieve, save_experiment,
                           snapshot["mmr_diversity"], {
                               "snapshot_id": snapshot_id,
                               "corpus_fingerprint": snapshot["corpus_fingerprint"],
                               "dataset_fingerprint": snapshot["dataset_fingerprint"],
                           })
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/benchmark/report")
def benchmark_report(dataset_name: str, strategy: str):
    """Return historical metrics and trends for one dataset/strategy pair."""
    if not dataset_name.strip() or len(dataset_name) > 120:
        raise HTTPException(status_code=400, detail="dataset_name must contain 1–120 characters.")
    if strategy not in ALLOWED_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"strategy must be one of: {', '.join(sorted(ALLOWED_STRATEGIES))}.")
    report = build_benchmark_report(list_experiments(100), dataset_name, strategy)
    if not report["run_count"]:
        raise HTTPException(status_code=404, detail="No batch benchmark history found for this dataset and strategy.")
    return report


@app.get("/api/benchmark/insights")
def benchmark_insights(dataset_name: str):
    """Return comparable latest strategy scorecards and deterministic recommendations."""
    if not dataset_name.strip() or len(dataset_name) > 120:
        raise HTTPException(status_code=400, detail="dataset_name must contain 1–120 characters.")
    return build_benchmark_insights(list_experiments(100), dataset_name)


@app.get("/api/benchmark/failure-analytics")
def benchmark_failure_analytics(dataset_name: str):
    if not dataset_name.strip() or len(dataset_name) > 120:
        raise HTTPException(status_code=400, detail="dataset_name must contain 1–120 characters.")
    return analyze_benchmark_failures(list_experiments(100), dataset_name)


@app.get("/api/benchmark/report.csv")
def benchmark_report_csv(dataset_name: str, strategy: str):
    """Download historical benchmark metrics as CSV."""
    report = benchmark_report(dataset_name, strategy)
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["run_id", "created_at", "dataset_name", "strategy", "case_count", *REPORT_METRICS])
    for run in report["history"]:
        writer.writerow([run["run_id"], run["created_at"], report["dataset_name"],
                         report["strategy"], run["case_count"],
                         *[run["metrics"][metric] for metric in REPORT_METRICS]])
    filename = f"benchmark-{dataset_name}-{strategy}.csv".replace(" ", "-")
    return Response(content=output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/api/quality/evaluate")
def evaluate_answer_quality(payload: dict):
    """Evaluate answer faithfulness against retrieved chunks and optional gold labels."""
    required = ("question", "answer", "retrieved_chunk_ids")
    if any(key not in payload for key in required):
        raise HTTPException(status_code=400, detail=f"Required fields: {', '.join(required)}")
    chunk_ids = payload["retrieved_chunk_ids"]
    relevant_ids = payload.get("relevant_chunk_ids", [])
    if not isinstance(chunk_ids, list) or not isinstance(relevant_ids, list):
        raise HTTPException(status_code=400, detail="Chunk ID fields must be arrays.")
    all_chunks = get_all_chunks()
    by_id = {item["id"]: item for item in all_chunks}
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in by_id]
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown retrieved chunk IDs: {missing}")
    contexts = [by_id[chunk_id] for chunk_id in chunk_ids]
    try:
        grounding = evaluate_answer_grounding(
            str(payload["question"]), str(payload["answer"]), contexts
        )
        result = {"grounding": grounding}
        if "relevant_chunk_ids" in payload:
            result["context_metrics"] = context_precision_recall(chunk_ids, relevant_ids)
        return result
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail="Gemini API key is required for faithfulness judging.") from exc
    except LLMGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
