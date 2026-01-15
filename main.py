"""
Unified FastAPI Application
Combines all backend services into a single application
"""

import os
import logging
import json
import time
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    BackgroundTasks,
    Header,
    WebSocket,
    WebSocketDisconnect,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, EmailStr, Field
from typing import List, Dict, Optional
from datetime import datetime

# Federated learning router and background worker
from api.routers.federated import router as federated_router, federated_worker
from api.routers.active_learning import router as active_learning_router

# Set default environment variables if not set (for calibration API)
if not os.getenv("CALIBRATION_JWT_SECRET"):
    os.environ["CALIBRATION_JWT_SECRET"] = "dev-secret-key-change-in-production"
if not os.getenv("CALIBRATION_JWT_AUDIENCE"):
    os.environ["CALIBRATION_JWT_AUDIENCE"] = "calibration-api"
if not os.getenv("CALIBRATION_JWT_ISSUER"):
    os.environ["CALIBRATION_JWT_ISSUER"] = "open-human-preference-modeling"

# Import from annotation interface
from annotation_interface.backend.models import (
    Task,
    Annotation,
    Annotator,
    QualityMetrics,
)
from annotation_interface.backend.quality_control import detect_spam
from annotation_interface.backend.auth import (
    register_user,
    login_user,
    get_user_by_id,
    validate_access_token,
    refresh_access_token,
    revoke_refresh_token,
    AuthError,
)

# Import calibration components
try:
    from calibration.auto_recalibration import (
        AutoRecalibrationService,
        ConvergenceFailedError,
        InsufficientDataError,
        RecalibrationInProgressError,
        ValidationDataError,
    )
    from common.auth import JWTBearer, JWTConfig
    from common.rate_limit import RateLimitExceeded, RateLimiter

    calibration_settings = AutoRecalibrationService()
    calibration_auth = JWTBearer(
        JWTConfig(
            secret=os.getenv("CALIBRATION_JWT_SECRET"),
            required_scope="calibration:write",
            audience=os.getenv("CALIBRATION_JWT_AUDIENCE"),
            issuer=os.getenv("CALIBRATION_JWT_ISSUER"),
        )
    )
    calibration_rate_limiter = RateLimiter(max_requests=10, window_seconds=3600)
    CALIBRATION_AVAILABLE = True
    # #region agent log
    import json
    import time

    log_entry = {
        "location": "main.py:calibration_import",
        "message": "Calibration loaded",
        "data": {},
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "D",
    }
    try:
        with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion
except Exception as e:
    CALIBRATION_AVAILABLE = False
    # #region agent log

    log_entry = {
        "location": "main.py:calibration_import",
        "message": "Calibration failed",
        "data": {"error": str(e), "error_type": type(e).__name__},
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "D",
    }
    try:
        with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion

# Import asyncio (needed for lifespan)
import asyncio

# Import monitoring components
try:
    import yaml
    from monitoring_dashboard.backend.models import Metric, Alert, AlertConfig
    from monitoring_dashboard.backend.metrics_collector import MetricsCollector
    from monitoring_dashboard.backend.alert_engine import AlertEngine

    metrics_collector = MetricsCollector()
    alert_configs = []
    try:
        with open("configs/monitoring_config.yaml", "r") as f:
            conf = yaml.safe_load(f)
            for r in conf.get("rules", []):
                alert_configs.append(
                    AlertConfig(
                        name=r["name"],
                        expr=r["expr"],
                        severity=r["severity"],
                        period_minutes=r["period_minutes"],
                        description=r["description"],
                    )
                )
    except Exception as e:
        logging.warning(f"Failed to load monitoring config: {e}")

    alert_engine = AlertEngine(alert_configs)
    MONITORING_AVAILABLE = True
    # #region agent log

    log_entry = {
        "location": "main.py:monitoring_import",
        "message": "Monitoring loaded",
        "data": {"alert_configs_count": len(alert_configs)},
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "D",
    }
    try:
        with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion
except Exception as e:
    logging.warning(f"Monitoring API not available: {e}")
    MONITORING_AVAILABLE = False
    # #region agent log

    log_entry = {
        "location": "main.py:monitoring_import",
        "message": "Monitoring failed",
        "data": {"error": str(e), "error_type": type(e).__name__},
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "D",
    }
    try:
        with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion

# Mock import for ActiveLearner
try:
    from active_learning.active_learner import ActiveLearner

    active_learner = ActiveLearner()
    active_learner.initialize_pools(20)
except ImportError:
    active_learner = None

# Import WebSocket manager
try:
    from common.websocket_manager import ws_manager, MessageType

    WEBSOCKET_AVAILABLE = True
    # #region agent log

    log_entry = {
        "location": "main.py:websocket_import",
        "message": "WebSocket loaded",
        "data": {"has_ws_manager": ws_manager is not None},
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "D",
    }
    try:
        with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion
except ImportError as e:
    logging.warning(f"WebSocket manager not available: {e}")
    WEBSOCKET_AVAILABLE = False
    ws_manager = None
    # #region agent log

    log_entry = {
        "location": "main.py:websocket_import",
        "message": "WebSocket failed",
        "data": {"error": str(e), "error_type": type(e).__name__},
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "D",
    }
    try:
        with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # #region agent log

    log_entry = {
        "location": "main.py:lifespan",
        "message": "App startup",
        "data": {
            "monitoring_available": MONITORING_AVAILABLE,
            "websocket_available": WEBSOCKET_AVAILABLE,
            "calibration_available": CALIBRATION_AVAILABLE,
            "has_ws_manager": ws_manager is not None,
        },
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "D",
    }
    try:
        with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion
    logger.info("Starting unified OpenHumanPreferenceModeling API")
    if MONITORING_AVAILABLE:
        asyncio.create_task(run_monitoring_poller())
    asyncio.create_task(federated_worker())
    if WEBSOCKET_AVAILABLE and ws_manager:
        await ws_manager.start_heartbeat()
    yield
    if WEBSOCKET_AVAILABLE and ws_manager:
        await ws_manager.stop_heartbeat()
    logger.info("Shutting down unified OpenHumanPreferenceModeling API")


async def run_monitoring_poller():
    """Background task for monitoring metrics collection."""
    while True:
        if MONITORING_AVAILABLE:
            metrics_collector.poll_all()
            all_metrics = metrics_collector.metrics_store
            fired_alerts = alert_engine.evaluate(all_metrics)

            # Broadcast updates via WebSocket
            if WEBSOCKET_AVAILABLE and ws_manager:
                # Group metrics by name for broadcast
                metrics_by_name = {}
                for m in all_metrics:
                    if m.name not in metrics_by_name:
                        metrics_by_name[m.name] = []
                    metrics_by_name[m.name].append(m)

                # Broadcast latest metrics
                for name, values in metrics_by_name.items():
                    if values:
                        latest = values[-1]
                        await ws_manager.broadcast_all(
                            MessageType.METRIC_UPDATE,
                            {
                                "name": name,
                                "value": latest.value,
                                "timestamp": latest.timestamp.isoformat() + "Z",
                                "tags": latest.tags,
                            },
                        )

                # Broadcast new alerts
                if fired_alerts:
                    for alert in fired_alerts:
                        await ws_manager.broadcast_all(
                            MessageType.ALERT_UPDATE,
                            {
                                "id": alert.id,
                                "rule_name": alert.rule_name,
                                "severity": alert.severity,
                                "status": alert.status,
                                "message": alert.message,
                                "timestamp": alert.timestamp.isoformat() + "Z",
                            },
                        )
        await asyncio.sleep(5)


# Create main app
app = FastAPI(
    title="OpenHumanPreferenceModeling API",
    description="Unified API for all backend services",
    version="1.0.0",
    lifespan=lifespan,
)

# API routers
app.include_router(federated_router)
app.include_router(active_learning_router)

# CORS middleware
cors_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]
# #region agent log

log_entry = {
    "location": "main.py:CORS",
    "message": "CORS config",
    "data": {"allowed_origins": cors_origins},
    "timestamp": int(time.time() * 1000),
    "sessionId": "debug-session",
    "runId": "run1",
    "hypothesisId": "B",
}
try:
    with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
except Exception:
    pass
# #endregion
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# #region agent log
@app.middleware("http")
async def log_requests(request: Request, call_next):
    log_entry = {
        "location": "main.py:middleware",
        "message": "Incoming request",
        "data": {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "origin": request.headers.get("origin"),
            "client_host": request.client.host if request.client else None,
        },
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "D",
    }
    try:
        with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    response = await call_next(request)
    log_entry2 = {
        "location": "main.py:middleware",
        "message": "Response sent",
        "data": {"status_code": response.status_code, "path": request.url.path},
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "D",
    }
    try:
        with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry2) + "\n")
    except Exception:
        pass
    return response


# #endregion


# Request/Response Models
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


class RecalibrationRequest(BaseModel):
    validation_data_uri: str
    target_ece: float = Field(default=0.08, ge=0.0, le=1.0)
    max_iterations: int = Field(default=100, ge=1, le=1000)


class RecalibrationResponse(BaseModel):
    temperature: float
    pre_ece: float
    post_ece: float
    iterations: int


class PredictionRecordRequest(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    correct: int = Field(ge=0, le=1)


class PredictRequest(BaseModel):
    state_vector: List[float] = Field(..., min_length=1, max_length=10)


@app.post("/api/predict")
async def api_predict(request: PredictRequest):
    """
    Simulate model prediction.
    Deterministic based on input vector sum.
    """
    # Simulate processing time
    await asyncio.sleep(0.1)

    # Deterministic logic
    # Use sum of vector as a seed for "randomness"
    vector_sum = sum(request.state_vector)

    # Generate probabilities for 3 fake actions
    raw_scores = [
        (vector_sum * 0.5) % 1.0,  # Action A
        (vector_sum * 0.3 + 0.2) % 1.0,  # Action B
        (vector_sum * 0.8 + 0.1) % 1.0,  # Action C
    ]

    # Softmax-ish normalization
    total = sum(raw_scores)
    probs = [s / total for s in raw_scores] if total > 0 else [0.33, 0.33, 0.34]

    # Pick max
    action_index = probs.index(max(probs))
    confidence = max(probs)

    return {
        "probabilities": probs,
        "action_index": action_index,
        "confidence": confidence,
    }


def _problem_detail(status: int, title: str, detail: str, code: str) -> JSONResponse:
    payload = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "code": code,
    }
    return JSONResponse(
        status_code=status, content=payload, media_type="application/problem+json"
    )


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return _problem_detail(
        status=422,
        title="Validation error",
        detail=str(exc),
        code="VALIDATION_ERROR",
    )


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
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization format")

        token = authorization[7:]
        payload = validate_access_token(token)

        user = get_user_by_id(payload["sub"])
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return UserResponse(**user.to_dict())
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))


# Dev Auth Bypass - ONLY enabled in development
DEV_MODE = os.getenv("OHPM_DEV_MODE", "false").lower() == "true"


class DevLoginRequest(BaseModel):
    user_id: str = "demo-user-id"


@app.post("/api/auth/dev-login", response_model=AuthTokensResponse)
async def api_dev_login(request: DevLoginRequest):
    """
    Development-only login bypass.

    This endpoint allows instant authentication without password
    for development and testing purposes. ONLY available when
    OHPM_DEV_MODE=true environment variable is set.

    DO NOT enable in production!
    """
    if not DEV_MODE:
        raise HTTPException(
            status_code=403,
            detail="Dev login is disabled. Set OHPM_DEV_MODE=true to enable.",
        )

    user = get_user_by_id(request.user_id)
    if not user:
        # Create a dev user if it doesn't exist
        from annotation_interface.backend.auth import users_db, User, hash_password

        password_hash, _ = hash_password("dev-password")
        user = User(
            id=request.user_id,
            email=f"{request.user_id}@dev.local",
            name=f"Dev User ({request.user_id})",
            password_hash=password_hash,
            role="admin",
        )
        users_db[user.id] = user

    # Generate tokens directly without password check
    from annotation_interface.backend.auth import generate_tokens

    tokens = generate_tokens(user.id)

    logger.info(f"Dev login: User {user.id} authenticated via dev bypass")

    return AuthTokensResponse(**tokens)


@app.get("/api/auth/dev-status")
async def api_dev_status():
    """Check if dev mode is enabled."""
    return {"devMode": DEV_MODE}


# ============================================================================
# Settings API
# ============================================================================

class SettingsRequest(BaseModel):
    company_name: Optional[str] = None
    company_phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    domain: Optional[str] = None
    allowed_file_types: Optional[str] = None
    site_direction: Optional[str] = Field(None, pattern="^(ltr|rtl)$")
    footer_info: Optional[str] = None


class SettingsResponse(BaseModel):
    company_name: str
    company_phone: str
    address: str
    city: str
    state: str
    zip_code: str
    domain: str
    allowed_file_types: str
    site_direction: str
    footer_info: str


# In-memory settings storage (in production, use a database)
settings_db: Dict[str, str] = {
    "company_name": "OpenHuman Preference Modeling",
    "company_phone": "",
    "address": "",
    "city": "",
    "state": "",
    "zip_code": "",
    "domain": "",
    "allowed_file_types": ".pdf, .csv, .fastq",
    "site_direction": "ltr",
    "footer_info": "{company_name}\n{address}\n{city}, {state} {zip}",
}


@app.get("/api/settings", response_model=SettingsResponse)
async def get_settings():
    """Get current application settings."""
    # #region agent log
    log_entry = {
        "location": "main.py:get_settings",
        "message": "Settings retrieved",
        "data": {},
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "A",
    }
    try:
        with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion
    return SettingsResponse(**settings_db)


@app.put("/api/settings", response_model=SettingsResponse)
async def update_settings(settings: SettingsRequest):
    """Update application settings."""
    # #region agent log
    log_entry = {
        "location": "main.py:update_settings",
        "message": "Settings updated",
        "data": {"fields_updated": [k for k, v in settings.model_dump(exclude_unset=True).items() if v is not None]},
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "A",
    }
    try:
        with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion
    
    # Update only provided fields
    update_data = settings.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            settings_db[key] = value
    
    return SettingsResponse(**settings_db)


# Health Check
@app.get("/api/health")
async def health_check(request: Request):
    """Health check endpoint."""
    # #region agent log

    log_entry = {
        "location": "main.py:health_check",
        "message": "Health check called",
        "data": {"client_host": request.client.host if request.client else None},
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "A",
    }
    try:
        with open("/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion
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
    """Get next high-priority task for annotator."""
    if active_learner:
        indices = active_learner.query_next(n=1, strategy_name="uncertainty")
        if indices:
            pass

    for t in tasks_db.values():
        if t.status == "unassigned":
            t.status = "assigned"
            t.assigned_to = annotator_id
            t.assigned_at = datetime.now()
            return t

    raise HTTPException(status_code=404, detail="No tasks available")


@app.post("/api/annotations")
async def submit_annotation(annotation: Annotation, background_tasks: BackgroundTasks):
    """Submit completed annotation."""
    if annotation.task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    annotations_db.append(annotation)

    task = tasks_db[annotation.task_id]
    task.status = "completed"

    background_tasks.add_task(update_learner, annotation)
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
        raise HTTPException(
            status_code=400, detail="page_size must be between 1 and 100"
        )

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
async def get_quality_metrics(annotator_id: str):
    """Get quality metrics for an annotator."""
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
        agreement_score=0.8,
        gold_pass_rate=0.9,
        avg_time_per_task=avg_time,
        flags=[],
    )


async def update_learner(annotation: Annotation):
    if active_learner:
        logger.debug("Updating learner with task: %s", annotation.task_id)


async def run_quality_checks(annotator_id: str):
    user_anns = [a for a in annotations_db if a.annotator_id == annotator_id]
    warnings = detect_spam(user_anns)
    if warnings:
        logger.warning("QC Warnings for %s: %s", annotator_id, warnings)


# Calibration Endpoints
if CALIBRATION_AVAILABLE:

    @app.post("/api/calibration/recalibrate", response_model=RecalibrationResponse)
    async def recalibrate(request: Request, payload: RecalibrationRequest):
        try:
            token_payload = calibration_auth(request.headers.get("Authorization"))
        except AuthError as exc:
            return _problem_detail(401, "Unauthorized", str(exc), "UNAUTHORIZED")

        subject = token_payload.get("sub") or request.client.host
        try:
            calibration_rate_limiter.enforce(str(subject))
        except RateLimitExceeded as exc:
            return _problem_detail(
                429, "Rate limit exceeded", str(exc), "RATE_LIMIT_EXCEEDED"
            )

        try:
            result = calibration_settings.runner.recalibrate(
                validation_data_uri=payload.validation_data_uri,
                target_ece=payload.target_ece,
                max_iterations=payload.max_iterations,
            )
        except InsufficientDataError as exc:
            return _problem_detail(
                400, "Insufficient data", str(exc), "INSUFFICIENT_DATA"
            )
        except ConvergenceFailedError as exc:
            return _problem_detail(
                500, "Convergence failed", str(exc), "CONVERGENCE_FAILED"
            )
        except ValidationDataError as exc:
            return _problem_detail(
                400, "Validation error", str(exc), "VALIDATION_ERROR"
            )
        except RecalibrationInProgressError as exc:
            return _problem_detail(
                409, "Recalibration in progress", str(exc), "VALIDATION_ERROR"
            )

        return RecalibrationResponse(
            temperature=result.temperature,
            pre_ece=result.pre_ece,
            post_ece=result.post_ece,
            iterations=result.iterations,
        )

    @app.post("/api/calibration/predictions")
    async def record_prediction(request: Request, payload: PredictionRecordRequest):
        try:
            calibration_auth(request.headers.get("Authorization"))
        except AuthError as exc:
            return _problem_detail(401, "Unauthorized", str(exc), "VALIDATION_ERROR")
        try:
            sampled = calibration_settings.monitor.record_prediction(
                payload.confidence, payload.correct
            )
        except ValidationDataError as exc:
            return _problem_detail(
                400, "Validation error", str(exc), "VALIDATION_ERROR"
            )
        return {"sampled": sampled}


# Monitoring Endpoints
if MONITORING_AVAILABLE:

    @app.get("/api/metrics", response_model=List[Metric])
    async def get_metrics(name: str):
        return metrics_collector.get_metrics(name)

    @app.get("/api/alerts", response_model=List[Alert])
    async def get_alerts():
        return alert_engine.get_alerts()

    @app.post("/api/alerts/{alert_id}/ack")
    async def ack_alert(alert_id: str):
        alert_engine.ack_alert(alert_id)
        return {"status": "success"}
else:
    # Return empty responses when monitoring is not available
    @app.get("/api/metrics")
    async def get_metrics(name: str):
        return []

    @app.get("/api/alerts")
    async def get_alerts():
        return []

    @app.post("/api/alerts/{alert_id}/ack")
    async def ack_alert(alert_id: str):
        return {"status": "success", "message": "Monitoring not available"}


# WebSocket Endpoint
if WEBSOCKET_AVAILABLE and ws_manager:

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket, token: str = Query(...)):
        """
        WebSocket endpoint for real-time event streaming.

        Connect with: ws://localhost:8000/ws/events?token=<access_token>

        Receives:
        - metric_update: Real-time metric changes
        - alert_update: New or updated alerts
        - task_assigned: New task assignments
        - calibration_status: Calibration progress
        - training_progress: Training run updates
        """
        # #region agent log

        log_entry = {
            "location": "main.py:websocket_events",
            "message": "WebSocket connection attempt",
            "data": {
                "has_token": bool(token),
                "origin": websocket.headers.get("origin", ""),
                "client_host": websocket.client.host if websocket.client else None,
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "E",
        }
        try:
            with open(
                "/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a"
            ) as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion
        # Validate token
        try:
            payload = validate_access_token(token)
            user_id = payload.get("sub", "anonymous")
            # #region agent log
            log_entry2 = {
                "location": "main.py:websocket_events",
                "message": "Token validated",
                "data": {"user_id": user_id},
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "E",
            }
            try:
                with open(
                    "/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a"
                ) as f:
                    f.write(json.dumps(log_entry2) + "\n")
            except Exception:
                pass
            # #endregion
        except AuthError as e:
            # #region agent log
            log_entry3 = {
                "location": "main.py:websocket_events",
                "message": "Token validation failed",
                "data": {"error": str(e)},
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "E",
            }
            try:
                with open(
                    "/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a"
                ) as f:
                    f.write(json.dumps(log_entry3) + "\n")
            except Exception:
                pass
            # #endregion
            await websocket.close(code=4001, reason=str(e))
            return

        # Check origin for security
        origin = websocket.headers.get("origin", "")
        allowed_origins = [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
        ]
        if origin and origin not in allowed_origins:
            await websocket.close(code=4003, reason="Origin not allowed")
            return

        # Connect and register
        connection_id = await ws_manager.connect(websocket, user_id)
        # #region agent log
        log_entry4 = {
            "location": "main.py:websocket_events",
            "message": "WebSocket connected",
            "data": {"connection_id": connection_id, "user_id": user_id},
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "E",
        }
        try:
            with open(
                "/Users/o11/OpenHumanPreferenceModeling/.cursor/debug.log", "a"
            ) as f:
                f.write(json.dumps(log_entry4) + "\n")
        except Exception:
            pass
        # #endregion

        try:
            while True:
                # Receive messages from client
                data = await websocket.receive_json()
                message_type = data.get("type")

                if message_type == "heartbeat_ack":
                    await ws_manager.handle_heartbeat_ack(connection_id)
                elif message_type == "sync_request":
                    # Handle sync request - send last known state
                    last_seq = data.get("payload", {}).get("lastSequence", 0)
                    # In a real implementation, we'd fetch missed messages from a store
                    await websocket.send_json(
                        {
                            "type": "sync_response",
                            "payload": {"synced": True, "fromSequence": last_seq},
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "sequence": -1,
                        }
                    )
                elif message_type == "join_room":
                    room = data.get("payload", {}).get("room")
                    if room:
                        await ws_manager.join_room(connection_id, room)
                elif message_type == "leave_room":
                    room = data.get("payload", {}).get("room")
                    if room:
                        await ws_manager.leave_room(connection_id, room)

        except WebSocketDisconnect:
            await ws_manager.disconnect(connection_id)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await ws_manager.disconnect(connection_id)

    @app.get("/api/ws/stats")
    async def websocket_stats():
        """Get WebSocket connection statistics."""
        return ws_manager.get_stats()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
