import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional, Any
from datetime import datetime
from uuid import uuid4

from .models import Task, Annotation, Annotator, PairwiseResponse, QualityMetrics
from .quality_control import detect_spam, compute_agreement
from .auth import (
    register_user,
    login_user,
    get_user_by_id,
    validate_access_token,
    refresh_access_token,
    revoke_refresh_token,
    AuthError,
)

# Configure logging
logger = logging.getLogger(__name__)

# Mock import for ActiveLearner since we might not have full env logic here
# In production this would import from active_learning.active_learner
try:
    from active_learning.active_learner import ActiveLearner

    active_learner = ActiveLearner()
    active_learner.initialize_pools(20)  # Small pool for demo
except ImportError:
    active_learner = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Annotation Interface API")
    yield
    logger.info("Shutting down Annotation Interface API")


app = FastAPI(title="Annotation Interface API", lifespan=lifespan)

# Optional dev log (off by default)
_DEBUG_LOG_PATH = os.getenv("OHPM_DEBUG_LOG_PATH")


def _debug_log(entry: Dict[str, Any]) -> None:
    if not _DEBUG_LOG_PATH:
        return
    import json
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        return


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    _debug_log(
        {
            "location": "annotation_interface.backend.main:middleware",
            "message": "Incoming request",
            "data": {
                "method": request.method,
                "path": request.url.path,
                "origin": request.headers.get("origin"),
                "client_host": request.client.host if request.client else None,
            },
            "timestamp": int(time.time() * 1000),
        }
    )
    response = await call_next(request)
    _debug_log(
        {
            "location": "annotation_interface.backend.main:middleware",
            "message": "Response sent",
            "data": {"status_code": response.status_code, "path": request.url.path},
            "timestamp": int(time.time() * 1000),
        }
    )
    return response


# Auth Request/Response Models
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str


class RefreshRequest(BaseModel):
    refreshToken: str


class AuthTokensResponse(BaseModel):
    accessToken: str
    refreshToken: str
    expiresIn: int
    tokenType: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    createdAt: str
    updatedAt: str


# Auth Endpoints
@app.post("/api/auth/register", response_model=AuthTokensResponse)
async def api_register(request: RegisterRequest):
    """Register a new user."""
    try:
        user, tokens = register_user(request.email, request.password, request.name)
        return AuthTokensResponse(**tokens)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login", response_model=AuthTokensResponse)
async def api_login(request: LoginRequest):
    """Login with email and password."""
    try:
        user, tokens = login_user(request.email, request.password)
        return AuthTokensResponse(**tokens)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/api/auth/refresh", response_model=AuthTokensResponse)
async def api_refresh(request: RefreshRequest):
    """Refresh access token."""
    try:
        tokens = refresh_access_token(request.refreshToken)
        return AuthTokensResponse(**tokens)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/api/auth/logout")
async def api_logout(request: RefreshRequest):
    """Logout and revoke refresh token."""
    revoke_refresh_token(request.refreshToken)
    return {"status": "success"}


@app.get("/api/auth/me", response_model=UserResponse)
async def api_me(authorization: Optional[str] = Header(None)):
    """Get current user info."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    try:
        # Extract Bearer token
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization format")

        token = authorization[7:]  # Remove "Bearer " prefix
        payload = validate_access_token(token)

        user = get_user_by_id(payload["sub"])
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return UserResponse(**user.to_dict())
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))


# Health Check
@app.get("/api/health")
async def health_check(request: Request):
    """Health check endpoint."""
    _debug_log(
        {
            "location": "annotation_interface.backend.main:health_check",
            "message": "Health check endpoint called",
            "data": {"client_host": request.client.host if request.client else None},
            "timestamp": int(__import__("time").time() * 1000),
        }
    )
    return {
        "encoder": "healthy",
        "dpo": "healthy",
        "monitoring": "healthy",
        "privacy": "healthy",
    }

# In-memory storage
tasks_db: Dict[str, Task] = {}
annotations_db: List[Annotation] = []
annotators_db: Dict[str, Annotator] = {}

# Pre-populate some tasks
for i in range(10):
    t = Task(
        type="pairwise",
        content={
            "prompt": f"Prompt {i}",
            "response_a": f"Response A for {i}",
            "response_b": f"Response B for {i}",
        },
        priority=0.5,
    )
    tasks_db[t.id] = t


@app.get("/api/tasks/next", response_model=Task)
async def get_next_task(annotator_id: str):
    """
    Get next high-priority task for annotator.
    """
    # 1. Check if learner has new queries
    if active_learner:
        indices = active_learner.query_next(n=1, strategy_name="uncertainty")
        if indices:
            idx = indices[0]
            # Convert to Task if not exists
            # For this mock, we just pick an unassigned from DB
            pass

    # Simple logic: find unassigned task
    for t in tasks_db.values():
        if t.status == "unassigned":
            # Lock it
            t.status = "assigned"
            t.assigned_to = annotator_id
            t.assigned_at = datetime.now()
            return t

    raise HTTPException(status_code=404, detail="No tasks available")


@app.post("/api/annotations")
async def submit_annotation(annotation: Annotation, background_tasks: BackgroundTasks):
    """
    Submit completed annotation.
    """
    # Validate task
    if annotation.task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    # Store
    annotations_db.append(annotation)

    # Update task status
    task = tasks_db[annotation.task_id]
    task.status = "completed"

    # Async: Update active learner (mock)
    background_tasks.add_task(update_learner, annotation)

    # Async: Run quality checks
    background_tasks.add_task(run_quality_checks, annotation.annotator_id)

    return {"status": "success", "id": annotation.id}


class PaginationMeta(BaseModel):
    page: int
    pageSize: int
    total: int
    totalPages: int
    hasNext: bool
    hasPrev: bool


class PaginatedAnnotations(BaseModel):
    data: List[Annotation]
    meta: PaginationMeta


@app.get("/api/annotations", response_model=PaginatedAnnotations)
async def list_annotations(
    annotator_id: Optional[str] = None,
    task_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """List annotations with optional filters and pagination."""
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")
    if page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="page_size must be between 1 and 100")

    filtered = annotations_db
    if annotator_id:
        filtered = [a for a in filtered if a.annotator_id == annotator_id]
    if task_id:
        filtered = [a for a in filtered if a.task_id == task_id]

    filtered = sorted(filtered, key=lambda a: a.created_at, reverse=True)

    total = len(filtered)
    total_pages = (total + page_size - 1) // page_size if total else 0
    start = (page - 1) * page_size
    end = start + page_size
    data = filtered[start:end] if start < total else []

    meta = PaginationMeta(
        page=page,
        pageSize=page_size,
        total=total,
        totalPages=total_pages,
        hasNext=page < total_pages,
        hasPrev=page > 1 and total_pages > 0,
    )
    return PaginatedAnnotations(data=data, meta=meta)


@app.get("/api/quality/metrics", response_model=QualityMetrics)
async def get_metrics(annotator_id: str):
    user_anns = [a for a in annotations_db if a.annotator_id == annotator_id]

    if not user_anns:
        return QualityMetrics(
            annotator_id=annotator_id,
            agreement_score=0.0,
            gold_pass_rate=0.0,
            avg_time_per_task=0.0,
        )

    avg_time = sum(a.time_spent_seconds for a in user_anns) / len(user_anns)

    return QualityMetrics(
        annotator_id=annotator_id,
        agreement_score=0.8,  # Mocked
        gold_pass_rate=0.9,  # Mocked
        avg_time_per_task=avg_time,
        flags=[],
    )


async def update_learner(annotation: Annotation):
    if active_learner:
        # In real implementation we'd map back to index
        logger.debug("Updating learner with task: %s", annotation.task_id)


async def run_quality_checks(annotator_id: str):
    user_anns = [a for a in annotations_db if a.annotator_id == annotator_id]
    warnings = detect_spam(user_anns)
    if warnings:
        logger.warning("QC Warnings for %s: %s", annotator_id, warnings)
