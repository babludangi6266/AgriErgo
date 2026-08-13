"""
FastAPI API Schemas — Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class ProcessingStatus(str, Enum):
    """Processing status states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoUploadResponse(BaseModel):
    """Response after uploading a video."""
    job_id: str
    filename: str
    file_size_mb: float
    status: ProcessingStatus = ProcessingStatus.PENDING
    message: str = "Video uploaded successfully"


class JobStatusResponse(BaseModel):
    """Response for job status check."""
    job_id: str
    status: ProcessingStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str = ""
    result_available: bool = False


class PostureDistribution(BaseModel):
    """Posture time distribution."""
    sitting_seconds: float = 0.0
    sitting_pct: float = 0.0
    standing_seconds: float = 0.0
    standing_pct: float = 0.0
    bending_seconds: float = 0.0
    bending_pct: float = 0.0
    walking_seconds: float = 0.0
    walking_pct: float = 0.0


class ErgonomicScoreResponse(BaseModel):
    """REBA ergonomic score."""
    reba_score: Optional[int] = None
    risk_level: Optional[str] = None
    action_required: Optional[str] = None


class WorkerSummaryResponse(BaseModel):
    """Summary of a single worker's assessment."""
    worker_id: int
    total_tracked_time: float
    posture: PostureDistribution
    load_events: int = 0
    repetitive_cycles_per_min: Optional[float] = None
    trip_count: int = 0
    tools_detected: List[str] = []
    longest_work_bout_seconds: float = 0.0
    total_rest_seconds: float = 0.0
    rest_count: int = 0
    ergonomic_score: ErgonomicScoreResponse = ErgonomicScoreResponse()


class VideoInfoResponse(BaseModel):
    """Video metadata."""
    filename: str
    duration_seconds: float
    resolution: str
    fps: float


class ProcessingResultResponse(BaseModel):
    """Complete processing result."""
    job_id: str
    video_info: VideoInfoResponse
    processing_time_seconds: float
    frames_processed: int
    workers_detected: int
    workers: List[WorkerSummaryResponse]
    full_report: Optional[Dict[str, Any]] = None
