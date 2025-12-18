import os
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from calibration.auto_recalibration import (
    AutoRecalibrationService,
    ConvergenceFailedError,
    InsufficientDataError,
    RecalibrationInProgressError,
    ValidationDataError,
)
from common.auth import AuthError, JWTBearer, JWTConfig
from common.rate_limit import RateLimitExceeded, RateLimiter


app = FastAPI(title="Calibration API")


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


def _problem_detail(status: int, title: str, detail: str, code: str) -> JSONResponse:
    payload = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "code": code,
    }
    return JSONResponse(status_code=status, content=payload, media_type="application/problem+json")


settings = AutoRecalibrationService()

jwt_secret = os.getenv("CALIBRATION_JWT_SECRET")
if not jwt_secret:
    raise RuntimeError("CALIBRATION_JWT_SECRET is required")
auth = JWTBearer(
    JWTConfig(
        secret=jwt_secret,
        required_scope="calibration:write",
        audience=os.getenv("CALIBRATION_JWT_AUDIENCE"),
        issuer=os.getenv("CALIBRATION_JWT_ISSUER"),
    )
)
rate_limiter = RateLimiter(max_requests=10, window_seconds=3600)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return _problem_detail(
        status=422,
        title="Validation error",
        detail=str(exc),
        code="VALIDATION_ERROR",
    )


@app.post("/api/calibration/recalibrate", response_model=RecalibrationResponse)
async def recalibrate(request: Request, payload: RecalibrationRequest):
    try:
        token_payload = auth(request.headers.get("Authorization"))
    except AuthError as exc:
        return _problem_detail(401, "Unauthorized", str(exc), "UNAUTHORIZED")

    subject = token_payload.get("sub") or request.client.host
    try:
        rate_limiter.enforce(str(subject))
    except RateLimitExceeded as exc:
        return _problem_detail(429, "Rate limit exceeded", str(exc), "RATE_LIMIT_EXCEEDED")

    try:
        result = settings.runner.recalibrate(
            validation_data_uri=payload.validation_data_uri,
            target_ece=payload.target_ece,
            max_iterations=payload.max_iterations,
        )
    except InsufficientDataError as exc:
        return _problem_detail(400, "Insufficient data", str(exc), "INSUFFICIENT_DATA")
    except ConvergenceFailedError as exc:
        return _problem_detail(500, "Convergence failed", str(exc), "CONVERGENCE_FAILED")
    except ValidationDataError as exc:
        return _problem_detail(400, "Validation error", str(exc), "VALIDATION_ERROR")
    except RecalibrationInProgressError as exc:
        return _problem_detail(409, "Recalibration in progress", str(exc), "VALIDATION_ERROR")

    return RecalibrationResponse(
        temperature=result.temperature,
        pre_ece=result.pre_ece,
        post_ece=result.post_ece,
        iterations=result.iterations,
    )


@app.post("/api/calibration/predictions")
async def record_prediction(request: Request, payload: PredictionRecordRequest):
    try:
        auth(request.headers.get("Authorization"))
    except AuthError as exc:
        return _problem_detail(401, "Unauthorized", str(exc), "VALIDATION_ERROR")
    try:
        sampled = settings.monitor.record_prediction(payload.confidence, payload.correct)
    except ValidationDataError as exc:
        return _problem_detail(400, "Validation error", str(exc), "VALIDATION_ERROR")
    return {"sampled": sampled}
