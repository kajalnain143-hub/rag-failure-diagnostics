from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)
    generate_answer: bool = True


class EvaluationRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    expected_answer: str = Field(min_length=2, max_length=5000)
    top_k: int = Field(default=4, ge=1, le=10)
