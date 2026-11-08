from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class JobRequest(BaseModel):
    operation: Literal["quantize", "benchmark", "inference"]
    checkpoint_id: str = Field(min_length=1, max_length=256)
    calibration_id: Optional[str] = None
    bits: Literal[4, 8] = 4
    group_size: Literal[32, 64, 128, 256] = 128
    device: Literal["cpu", "cuda"] = "cpu"
    prompt: Optional[str] = Field(default=None, max_length=4096)
    image: Optional[str] = None
    trusted_checkpoint: bool = Field(default=False, strict=True)
    max_new_tokens: int = Field(default=64, ge=1, le=128)

    @model_validator(mode="after")
    def validate_inference_request(self):
        if self.operation == "inference" and not self.prompt:
            raise ValueError("A non-empty text prompt is required for inference.")
        return self


class JobLog(BaseModel):
    timestamp: datetime
    level: Literal["INFO", "WARNING", "ERROR", "SUCCESS"]
    message: str


class JobRecord(BaseModel):
    id: str
    operation: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: Optional[int] = None
    request: dict[str, Any]
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    logs: list[JobLog] = Field(default_factory=list)


class CheckpointRecord(BaseModel):
    id: str
    name: str
    path: str
    format: str
    precision: str
    size_bytes: int
    parameters: Optional[int] = None
    lineage: str
    multimodal: bool
    available: bool = True


class JobSubmission(BaseModel):
    job_id: str
    status: str
    message: str


class ImportResultsRequest(BaseModel):
    manifest: dict[str, Any]
    source_label: Optional[str] = Field(default=None, max_length=256)
