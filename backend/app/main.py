from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from .database import get_all_chunks, get_documents, init_db, save_document
from .diagnostics import diagnose, evaluate_retrieval
from .llm import LLMGenerationError, LLMNotConfiguredError, generate_grounded_answer
from .embeddings import EMBEDDING_MODEL, embed_query, embed_text
from .failure_clinic import PATTERNS, diagnose_incident
from .incident_schemas import IncidentDiagnosisRequest
from .rag_engine import retrieve, split_into_chunks
from .schemas import EvaluationRequest, QueryRequest
from .experiment_store import init_experiment_store, list_experiments, save_experiment
from .evaluation_metrics import evaluate_case

app = FastAPI(
    title="RAG Failure Diagnostics API",
    description="A RAG workbench for grounded answers, retrieval diagnostics, and evaluation.",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    init_experiment_store()


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
            return "
".join(page.extract_text() or "" for page in reader.pages)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail="The PDF could not be parsed.") from exc
    raise HTTPException(status_code=400, detail="Only PDF, TXT, and Markdown files are supported.")


@app.get("/")
def read_root():
    return {"message": "RAG Failure Diagnostics API is running.", "version": app.version, "docs": "/docs"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "rag-failure-diagnostics-api"}


@app.get("/api/documents")
def list_documents():
    return {"documents": get_documents()}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "uploaded-document").name
    content = await file.read()
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
    chunks = get_all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="Upload a document before asking a question.")
    query_vector = embed_query(request.question)
    results = retrieve(request.question, chunks, request.top_k, query_embedding=query_vector)
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
        try:
            answer, answer_mode = generate_grounded_answer(request.question, results), "gemini"
        except LLMNotConfiguredError:
            answer, answer_mode = "Gemini API key is not configured. Add GEMINI_API_KEY to backend/.env to enable generation.", "llm_not_configured"
        except LLMGenerationError as exc:
            answer, answer_mode = str(exc), "llm_error"
    return {"question": request.question, "answer": answer, "answer_mode": answer_mode,
            "model": "gemini-2.5-flash" if answer_mode == "gemini" else None,
            "retrieved_chunks": results, "sources": sources, "diagnostics": diagnostic, "retrieval_mode": results[0].get("retrieval_mode") if results else "none"}


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
        return {
            "incident": {
                "description": request.description,
                "expected_behavior": request.expected_behavior,
            },
            "diagnosis": diagnosis,
        }
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail="Gemini API key is not configured.") from exc
    except LLMGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
