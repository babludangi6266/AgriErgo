"""
FastAPI Backend — Video upload, processing, and result serving.

Endpoints:
    POST /api/upload          — Upload a video file
    POST /api/process/{id}    — Trigger processing
    GET  /api/status/{id}     — Check processing status
    GET  /api/results/{id}    — Get structured results (JSON)
    GET  /api/results/{id}/csv — Download CSV report
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import UPLOADS_DIR, RESULTS_DIR, SUPPORTED_FORMATS, MAX_UPLOAD_SIZE_MB
from api.schemas import (
    VideoUploadResponse,
    JobStatusResponse,
    ProcessingResultResponse,
    VideoInfoResponse,
    WorkerSummaryResponse,
    PostureDistribution,
    ErgonomicScoreResponse,
    ProcessingStatus,
)
from api.dependencies import job_store

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="AgriErgo API",
    description="Video-Based Farm Worker Ergonomics & Drudgery Assessment Platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "platform": "AgriErgo v0.1.0"}


@app.post("/api/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for processing.

    Accepts: MP4, AVI, MOV, MKV, WMV
    Max size: 2 GB
    """
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {SUPPORTED_FORMATS}",
        )

    # Save the uploaded file
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOADS_DIR / file.filename
    file_size = 0

    with open(file_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
            file_size += len(chunk)
            if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                os.remove(file_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB} MB",
                )
            buffer.write(chunk)

    # Create a job
    job = job_store.create_job(
        filename=file.filename,
        file_path=str(file_path),
        file_size_mb=file_size / (1024 * 1024),
    )

    return VideoUploadResponse(
        job_id=job.job_id,
        filename=job.filename,
        file_size_mb=job.file_size_mb,
        status=ProcessingStatus.PENDING,
    )


def _run_pipeline(job_id: str):
    """Background task: run the processing pipeline."""
    from agriergo.pipeline import AgriErgoPipeline

    job = job_store.get_job(job_id)
    if not job:
        return

    job_store.update_job(job_id, status=ProcessingStatus.PROCESSING, progress=0.0)

    def progress_callback(progress: float, message: str):
        job_store.update_job(job_id, progress=progress, message=message)

    try:
        pipeline = AgriErgoPipeline()
        result = pipeline.process(job.file_path, progress_callback=progress_callback)

        job_store.update_job(
            job_id,
            status=ProcessingStatus.COMPLETED,
            progress=1.0,
            message=f"Completed: {result.workers_detected} workers, "
                    f"{result.frames_processed} frames in {result.processing_time_seconds}s",
            result=result,
        )
    except Exception as e:
        job_store.update_job(
            job_id,
            status=ProcessingStatus.FAILED,
            message=str(e),
            error=str(e),
        )


@app.post("/api/process/{job_id}")
async def process_video(job_id: str, background_tasks: BackgroundTasks):
    """Trigger video processing for a previously uploaded video."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == ProcessingStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="Job already processing")

    background_tasks.add_task(_run_pipeline, job_id)

    return {"job_id": job_id, "status": "processing", "message": "Processing started"}


@app.get("/api/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """Check the processing status of a job."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=round(job.progress, 2),
        message=job.message,
        result_available=job.result is not None,
    )


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get structured JSON results for a completed job."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != ProcessingStatus.COMPLETED or job.result is None:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job.status.value}",
        )

    result = job.result
    return JSONResponse(content=result.json_report)


@app.get("/api/results/{job_id}/csv")
async def get_results_csv(job_id: str):
    """Download CSV report for a completed job."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != ProcessingStatus.COMPLETED or job.result is None:
        raise HTTPException(status_code=400, detail="Job not completed")

    result = job.result
    return Response(
        content=result.csv_report,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={Path(job.filename).stem}_report.csv"
        },
    )


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    jobs = job_store.list_jobs()
    return [
        {
            "job_id": j.job_id,
            "filename": j.filename,
            "status": j.status.value,
            "progress": j.progress,
            "message": j.message,
        }
        for j in jobs
    ]
