from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import yaml
from active_learning.active_learner import ActiveLearner
from active_learning.dependencies import get_active_learner

router = APIRouter(prefix="/api/active-learning", tags=["active-learning"])


def get_learner():
    return get_active_learner()


class ALConfig(BaseModel):
    budget: int
    batch_size: int
    seed_size: int
    strategy: Optional[str] = "iid"


class ALStatus(BaseModel):
    labeledCount: int
    unlabeledCount: int
    budgetRemaining: int
    currentStrategy: str
    lastUpdated: str


class QueueItem(BaseModel):
    id: str
    text: str
    uncertaintyScore: float
    diversityScore: float
    iidScore: float
    compositeRank: int
    createdAt: str


@router.get("/config", response_model=ALConfig)
def get_config(learner: ActiveLearner = Depends(get_learner)):
    return {
        "budget": learner.budget,
        "batch_size": learner.batch_size,
        "seed_size": 100,  # This is static in current implementation but could be dynamic
        "strategy": "iid",  # Default for now, learner needs to track this
    }


@router.patch("/config")
def update_config(config: ALConfig, learner: ActiveLearner = Depends(get_learner)):
    learner.budget = config.budget
    learner.batch_size = config.batch_size
    # Update YAML file
    try:
        with open("configs/active_learning_config.yaml", "r") as f:
            yaml_config = yaml.safe_load(f)

        yaml_config["active_learning"]["budget"] = config.budget
        yaml_config["active_learning"]["batch_size"] = config.batch_size
        yaml_config["active_learning"]["seed_size"] = config.seed_size

        with open("configs/active_learning_config.yaml", "w") as f:
            yaml.dump(yaml_config, f)

        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")


@router.get("/status", response_model=ALStatus)
def get_status(learner: ActiveLearner = Depends(get_learner)):
    from datetime import datetime

    return {
        "labeledCount": len(learner.labeled_pool),
        "unlabeledCount": len(learner.unlabeled_pool),
        "budgetRemaining": learner.budget,
        "currentStrategy": "iid",  # Placeholder, learner should track
        "lastUpdated": datetime.now().isoformat(),
    }


@router.get("/queue", response_model=List[QueueItem])
def get_queue(learner: ActiveLearner = Depends(get_learner)):
    # Get next suggestions
    indices = learner.query_next(n=20, strategy_name="iid")

    # Map to QueueItem model
    queue_items = []
    for rank, idx in enumerate(indices):
        item = learner.unlabeled_pool[idx]
        # Calculate scores (in real implementation, these would come from the model/strategy)
        queue_items.append(
            {
                "id": str(item.get("id", idx)),
                "text": item.get("text", f"Sample {idx}"),
                "uncertaintyScore": float(item.get("uncertainty", 0.5)),  # Mock
                "diversityScore": float(item.get("diversity", 0.5)),  # Mock
                "iidScore": float(item.get("iid", 0.5)),  # Mock
                "compositeRank": rank + 1,
                "createdAt": "2024-01-01T00:00:00Z",  # Mock
            }
        )
    return queue_items


@router.post("/refresh")
def refresh_predictions(learner: ActiveLearner = Depends(get_learner)):
    # Trigger a retraining or re-ranking
    # For now, just a placeholder as query_next does the heavy lifting
    return {"status": "refreshed"}
