"""
Authentication Backend Module

Purpose: User authentication with JWT tokens
"""

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

from common.auth import generate_jwt_hs256, decode_jwt_hs256, AuthError


# In-memory user store (replace with database in production)
users_db: Dict[str, "User"] = {}
refresh_tokens_db: Dict[str, str] = {}  # token -> user_id


@dataclass
class User:
    id: str
    email: str
    name: str
    password_hash: str
    role: str = "annotator"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Hash password with salt using SHA-256."""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}${hashed}", salt


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against stored hash."""
    try:
        salt, hashed = password_hash.split("$")
        new_hash, _ = hash_password(password, salt)
        return new_hash == password_hash
    except ValueError:
        return False


# JWT configuration
JWT_SECRET = os.getenv("AUTH_JWT_SECRET", "dev-secret-change-in-production")
JWT_ACCESS_EXPIRY = 3600  # 1 hour
JWT_REFRESH_EXPIRY = 86400 * 7  # 7 days


def generate_tokens(user_id: str) -> dict:
    """Generate access and refresh tokens for a user."""
    now = int(time.time())

    access_payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + JWT_ACCESS_EXPIRY,
        "type": "access",
    }

    refresh_token = secrets.token_urlsafe(32)
    refresh_tokens_db[refresh_token] = user_id

    access_token = generate_jwt_hs256(access_payload, JWT_SECRET)

    return {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "expiresIn": JWT_ACCESS_EXPIRY,
        "tokenType": "Bearer",
    }


def validate_access_token(token: str) -> dict:
    """Validate access token and return payload."""
    try:
        payload = decode_jwt_hs256(token, JWT_SECRET)

        # Check expiration
        if payload.get("exp", 0) < time.time():
            raise AuthError("Token expired")

        # Check type
        if payload.get("type") != "access":
            raise AuthError("Invalid token type")

        return payload
    except Exception as e:
        raise AuthError(str(e))


def refresh_access_token(refresh_token: str) -> dict:
    """Generate new access token from refresh token."""
    user_id = refresh_tokens_db.get(refresh_token)
    if not user_id:
        raise AuthError("Invalid refresh token")

    # Remove old refresh token
    del refresh_tokens_db[refresh_token]

    # Generate new tokens
    return generate_tokens(user_id)


def revoke_refresh_token(refresh_token: str) -> None:
    """Revoke a refresh token."""
    refresh_tokens_db.pop(refresh_token, None)


def register_user(email: str, password: str, name: str) -> tuple[User, dict]:
    """Register a new user."""
    # Check if email already exists
    for user in users_db.values():
        if user.email == email:
            raise ValueError("Email already registered")

    # Validate password strength
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")

    # Create user
    user_id = str(uuid4())
    password_hash, _ = hash_password(password)

    user = User(
        id=user_id,
        email=email,
        name=name,
        password_hash=password_hash,
    )

    users_db[user_id] = user

    # Generate tokens
    tokens = generate_tokens(user_id)

    return user, tokens


def login_user(email: str, password: str) -> tuple[User, dict]:
    """Authenticate user and return tokens."""
    # Find user by email
    user = None
    for u in users_db.values():
        if u.email == email:
            user = u
            break

    if not user:
        raise AuthError("Invalid email or password")

    if not verify_password(password, user.password_hash):
        raise AuthError("Invalid email or password")

    # Generate tokens
    tokens = generate_tokens(user.id)

    return user, tokens


def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by ID."""
    return users_db.get(user_id)


# Create demo user for development
def create_demo_user():
    """Create a demo user for testing."""
    if not any(u.email == "demo@example.com" for u in users_db.values()):
        password_hash, _ = hash_password("Demo1234")
        demo_user = User(
            id="demo-user-id",
            email="demo@example.com",
            name="Demo User",
            password_hash=password_hash,
            role="admin",
        )
        users_db[demo_user.id] = demo_user


# Initialize demo user
create_demo_user()
