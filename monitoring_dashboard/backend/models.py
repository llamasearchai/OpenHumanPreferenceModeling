from typing import List, Optional, Any, Dict
from pydantic import BaseModel
from datetime import datetime


class Metric(BaseModel):
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = {}


class AlertConfig(BaseModel):
    name: str
    expr: str
    severity: str
    period_minutes: int
    description: str


class Alert(BaseModel):
    id: str
    rule_name: str
    severity: str
    status: str  # pending, firing, resolved, acknowledged
    timestamp: datetime
    message: str
