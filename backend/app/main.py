from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from .database import get_all_chunks, get_documents, init_db, save_document
from .diagnostics import diagnose, evaluate_retrieval
from .rag_engine import retrieve, split_into_chunks
from .schemas import EvaluationRequest, QueryRequest

app = FastAPI(
    title="RAG Failure Diagnostics API",
    description="A lightweight platform for inspecting and evaluating RAG retrieval failures.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise HTTPException(status_code=400, detail="Only PDF, TXT, and Markdown files are supported.")


@app.get("/")
def read_root():
    return {
        "message": "RAG Failure Diagnostics API is running.",
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "rag-failure-diagnostics-api"}


@app.get("/api/documents")
def list_documents():
    return {"documents": get_documents()}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded-document"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    text = extract_text(filename, content)
    chunks = split_into_chunks(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text was found in the document.")

    document_id = save_document(filename, chunks)
    return {
        "message": "Document uploaded and indexed successfully.",
        "document_id": document_id,
        "filename": filename,
        "chunk_count": len(chunks),
    }


@app.post("/api/query")
def query_documents(request: QueryRequest):
    chunks = get_all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="Upload a document before asking a question.")

    results = retrieve(request.question, chunks, request.top_k)
    diagnostic = diagnose(request.question, results)
    answer = (
        results[0]["content"][:700]
        if results and results[0]["score"] >= 0.05
        else "I could not find sufficiently relevant context in the indexed documents."
    )
    return {
        "question": request.question,
        "answer": answer,
        "retrieved_chunks": results,
        "diagnostics": diagnostic,
    }


@app.post("/api/evaluate")
def evaluate(request: EvaluationRequest):
    chunks = get_all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="Upload a document before running evaluation.")

    results = retrieve(request.question, chunks, request.top_k)
    return {
        "evaluation": evaluate_retrieval(request.question, request.expected_answer, results),
        "diagnostics": diagnose(request.question, results),
        "retrieved_chunks": results,
    }
