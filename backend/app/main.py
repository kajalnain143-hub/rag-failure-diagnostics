from fastapi import FastAPI

app = FastAPI(
    title="RAG Failure Diagnostics API",
    description="An API for diagnosing and evaluating Retrieval-Augmented Generation systems.",
    version="0.1.0",
)


@app.get("/")
def read_root():
    """Return a welcome message and basic API information."""
    return {
        "message": "RAG Failure Diagnostics API is running.",
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    """Provide a simple health-check endpoint."""
    return {
        "status": "healthy",
        "service": "rag-failure-diagnostics-api",
    }
