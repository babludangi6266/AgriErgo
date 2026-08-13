"""
FastAPI Dependencies — Shared state and utilities for the API.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from api.schemas import ProcessingStatus


@dataclass
class Job:
    """Represents a video processing job."""
    job_id: str
    filename: str
    file_path: str
    file_size_mb: float
    status: ProcessingStatus = ProcessingStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None


class JobStore:
    """
    In-memory job store for the prototype.
    Phase 2 will replace with Redis/DB-backed store.
    """

    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    def create_job(self, filename: str, file_path: str, file_size_mb: float) -> Job:
        """Create a new job and return it."""
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            job_id=job_id,
            filename=filename,
            file_path=file_path,
            file_size_mb=round(file_size_mb, 2),
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        """Update job fields."""
        job = self._jobs.get(job_id)
        if job:
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)

    def list_jobs(self):
        """List all jobs."""
        return list(self._jobs.values())


# Global singleton for the prototype
job_store = JobStore()
