import random
import asyncio
from datetime import datetime
from typing import List, Optional, Dict
import hashlib
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from privacy_engine.federated_learning.coordinator import Coordinator

router = APIRouter(prefix="/api/federated", tags=["Federated Learning"])

# --- Models ---


class PrivacyBudget(BaseModel):
    epsilonSpent: float
    epsilonRemaining: float
    delta: float
    totalSteps: int


class FederatedStatus(BaseModel):
    round: int
    isActive: bool
    totalClients: int
    activeClients: int
    privacyBudget: PrivacyBudget
    modelChecksum: str
    lastUpdated: str


class GradientStats(BaseModel):
    meanNorm: float
    maxNorm: float
    noiseScale: float


class RoundDetails(BaseModel):
    roundId: int
    startedAt: str
    completedAt: Optional[str]
    participatingClients: int
    selectedClients: int
    gradientStats: GradientStats
    status: str  # 'in_progress', 'completed', 'failed'


class ClientParticipation(BaseModel):
    clientId: str
    rounds: List[int]
    lastSeen: str
    status: str  # 'active', 'straggler', 'offline'


# --- Coordinator Service (Stateful) ---


class FederatedService:
    def __init__(self):
        self.coordinator = Coordinator(num_clients=100)
        self.is_active = False
        self.active_clients_count = 87
        self.last_updated = datetime.now()
        self.model_checksum = self._calculate_checksum()

        # History storage
        self.rounds_history: List[RoundDetails] = []
        self.clients_db: Dict[str, ClientParticipation] = self._init_clients()

        # Background task control
        self._loop_task = None

    def _calculate_checksum(self) -> str:
        # Simple checksum of the model parameters
        model_bytes = self.coordinator.global_model.tobytes()
        return hashlib.md5(model_bytes).hexdigest()[:12]

    def _init_clients(self) -> Dict[str, ClientParticipation]:
        clients = {}
        for i in range(100):
            cid = f"client-{i:03d}"
            status = "active"
            if random.random() > 0.9:
                status = "offline"
            elif random.random() > 0.8:
                status = "straggler"

            clients[cid] = ClientParticipation(
                clientId=cid,
                rounds=[],
                lastSeen=datetime.now().isoformat(),
                status=status,
            )
        return clients

    def get_status(self) -> FederatedStatus:
        pb_status = self.coordinator.privacy_tracker.current_status()
        # pb_status example: {'epsilon': 0.5, 'delta': 1e-5}
        # We need to map/mock remaining
        epsilon_spent = pb_status.get("epsilon", 0.0)

        return FederatedStatus(
            round=self.coordinator.round_num,
            isActive=self.is_active,
            totalClients=self.coordinator.num_clients,
            activeClients=self.active_clients_count,
            privacyBudget=PrivacyBudget(
                epsilonSpent=epsilon_spent,
                epsilonRemaining=10.0 - epsilon_spent,  # Assuming budget 10
                delta=pb_status.get("delta", 1e-5),
                totalSteps=self.coordinator.round_num * 100,  # Mock steps
            ),
            modelChecksum=self.model_checksum,
            lastUpdated=self.last_updated.isoformat(),
        )

    def start_training(self):
        self.is_active = True
        self.last_updated = datetime.now()

    def pause_training(self):
        self.is_active = False
        self.last_updated = datetime.now()

    def get_rounds(self) -> List[RoundDetails]:
        return sorted(self.rounds_history, key=lambda r: r.roundId, reverse=True)

    def get_clients(self) -> List[ClientParticipation]:
        return list(self.clients_db.values())

    async def run_round_logic(self):
        """Simulate a single round execution."""
        if not self.is_active:
            return

        current_round_id = self.coordinator.round_num + 1

        # Create In-Progress Round Entry
        round_entry = RoundDetails(
            roundId=current_round_id,
            startedAt=datetime.now().isoformat(),
            completedAt=None,
            participatingClients=0,
            selectedClients=int(
                self.coordinator.num_clients * self.coordinator.fraction_fit
            ),
            gradientStats=GradientStats(meanNorm=0, maxNorm=0, noiseScale=0),
            status="in_progress",
        )
        self.rounds_history.insert(0, round_entry)

        # Call internal coordinator
        selected_clients, _ = self.coordinator.start_round()

        # Simulate delay for training
        await asyncio.sleep(2)

        # Simulate aggregation
        gradients = [np.random.randn(100).tolist() for _ in selected_clients]
        self.coordinator.aggregate_gradients(gradients)

        # Update round entry completion
        round_entry.status = "completed"
        round_entry.completedAt = datetime.now().isoformat()
        round_entry.participatingClients = len(selected_clients)
        round_entry.gradientStats = GradientStats(
            meanNorm=float(np.mean([np.linalg.norm(g) for g in gradients])),
            maxNorm=float(np.max([np.linalg.norm(g) for g in gradients])),
            noiseScale=0.1,
        )

        # Update checksum
        self.model_checksum = self._calculate_checksum()
        self.last_updated = datetime.now()

        # Update clients
        for cid_idx in selected_clients:
            cid = f"client-{cid_idx:03d}"
            if cid in self.clients_db:
                self.clients_db[cid].rounds.append(current_round_id)
                self.clients_db[cid].lastSeen = datetime.now().isoformat()


# Singleton instance
fed_service = FederatedService()

# --- Background Worker ---


async def federated_worker():
    """Background loop to run rounds if active."""
    while True:
        if fed_service.is_active:
            await fed_service.run_round_logic()
        await asyncio.sleep(5)  # Period between rounds


# --- Endpoints ---


@router.get("/status", response_model=FederatedStatus)
async def get_status():
    return fed_service.get_status()


@router.get("/rounds", response_model=List[RoundDetails])
async def get_rounds():
    return fed_service.get_rounds()


@router.get("/clients", response_model=List[ClientParticipation])
async def get_clients():
    return fed_service.get_clients()


@router.post("/start")
async def start_training():
    fed_service.start_training()
    return {"status": "started"}


@router.post("/pause")
async def pause_training():
    fed_service.pause_training()
    return {"status": "paused"}
