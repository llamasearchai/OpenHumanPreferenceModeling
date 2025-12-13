import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


class AuthError(Exception):
    pass


@dataclass
class JWTConfig:
    secret: str
    required_scope: str
    audience: Optional[str] = None
    issuer: Optional[str] = None


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise AuthError("Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("Invalid Authorization header")
    return parts[1]


def decode_jwt_hs256(token: str, secret: str) -> Dict[str, Any]:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError as exc:
        raise AuthError("Malformed JWT") from exc

    signed = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).digest()
    signature = _b64url_decode(signature_b64)

    if not hmac.compare_digest(signature, expected_sig):
        raise AuthError("Invalid JWT signature")

    header = json.loads(_b64url_decode(header_b64))
    if header.get("alg") != "HS256":
        raise AuthError("Unsupported JWT algorithm")

    payload = json.loads(_b64url_decode(payload_b64))
    return payload


def validate_claims(
    payload: Dict[str, Any],
    audience: Optional[str],
    issuer: Optional[str],
    now: Optional[float] = None,
) -> None:
    now_ts = now if now is not None else time.time()
    if audience and payload.get("aud") != audience:
        raise AuthError("Invalid audience")
    if issuer and payload.get("iss") != issuer:
        raise AuthError("Invalid issuer")
    exp = payload.get("exp")
    if exp is not None and now_ts > float(exp):
        raise AuthError("Token expired")


def extract_scopes(payload: Dict[str, Any]) -> Iterable[str]:
    scopes = payload.get("scope") or payload.get("scopes") or []
    if isinstance(scopes, str):
        return scopes.split()
    if isinstance(scopes, list):
        return [str(scope) for scope in scopes]
    return []


def require_scope(payload: Dict[str, Any], required_scope: str) -> None:
    scopes = set(extract_scopes(payload))
    if required_scope not in scopes:
        raise AuthError("Missing required scope")


def generate_jwt_hs256(payload: Dict[str, Any], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload).encode("utf-8"))
    signed = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).digest()
    signature_b64 = _b64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


class JWTBearer:
    def __init__(self, config: JWTConfig):
        self.config = config

    def __call__(self, authorization_header: Optional[str]) -> Dict[str, Any]:
        token = extract_bearer_token(authorization_header)
        payload = decode_jwt_hs256(token, self.config.secret)
        validate_claims(payload, self.config.audience, self.config.issuer)
        require_scope(payload, self.config.required_scope)
        return payload
