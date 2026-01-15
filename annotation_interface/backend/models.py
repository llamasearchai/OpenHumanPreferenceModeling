from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["pairwise", "ranking", "critique", "likert"]
    content: Dict[str, Any]  # e.g. {"prompt": "...", "responses": [...]}
    created_at: datetime = Field(default_factory=datetime.now)
    priority: float = 0.0
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    status: Literal["unassigned", "assigned", "completed"] = "unassigned"


class Annotation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    annotator_id: str
    annotation_type: str
    response_data: Dict[str, Any]
    time_spent_seconds: float
    confidence: int = Field(ge=1, le=5)
    created_at: datetime = Field(default_factory=datetime.now)


class PairwiseResponse(BaseModel):
    winner: Literal["A", "B", "tie", "both_poor"]
    rationale: Optional[str] = Field(max_length=500)


class Annotator(BaseModel):
    id: str
    skill_level: int = 1
    total_annotations: int = 0
    accuracy: float = 0.0
    status: Literal["active", "probation", "suspended"] = "active"


class QualityMetrics(BaseModel):
    annotator_id: str
    agreement_score: float
    gold_pass_rate: float
    avg_time_per_task: float
    flags: List[str] = []
