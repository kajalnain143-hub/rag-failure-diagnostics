from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from .database import get_all_chunks, get_documents, init_db, save_document
from .diagnostics import diagnose, evaluate_retrieval
from .llm import LLMGenerationError, LLMNotConfiguredError, generate_grounded_answer
from .rag_engine import retrieve, split_into_chunks
from .schemas import EvaluationRequest, QueryRequest

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
    document_id = save_document(filename, chunks)
    return {"message": "Document uploaded and indexed successfully.", "document_id": document_id,
            "filename": filename, "chunk_count": len(chunks)}


@app.post("/api/query")
def query_documents(request: QueryRequest):
    chunks = get_all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="Upload a document before asking a question.")
    results = retrieve(request.question, chunks, request.top_k)
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
            "retrieved_chunks": results, "sources": sources, "diagnostics": diagnostic}


@app.post("/api/evaluate")
def evaluate(request: EvaluationRequest):
    chunks = get_all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="Upload a document before running evaluation.")
    results = retrieve(request.question, chunks, request.top_k)
    return {"evaluation": evaluate_retrieval(request.question, request.expected_answer, results),
            "diagnostics": diagnose(request.question, results), "retrieved_chunks": results}
