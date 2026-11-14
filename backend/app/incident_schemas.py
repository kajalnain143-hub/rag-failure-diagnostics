from pydantic import BaseModel, Field


class IncidentDiagnosisRequest(BaseModel):
    description: str = Field(min_length=10, max_length=12000)
    logs: str = Field(default="", max_length=20000)
    expected_behavior: str = Field(default="", max_length=5000)
