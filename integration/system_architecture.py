import asyncio
from typing import Dict, Any

# Import mocked components
# In a real system, these would be gRPC stubs connecting to remote services.
# Here, we import the Python classes directly to simulate the "System".

from user_state_encoder.encoder import UserStateEncoder
from dpo_pipeline.preference_data_generator import PreferenceDataGenerator
from monitoring_dashboard.backend.metrics_collector import MetricsCollector
from monitoring_dashboard.backend.alert_engine import AlertEngine, AlertConfig
from privacy_engine.privacy_budget_tracker import PrivacyBudgetTracker


class SystemOrchestrator:
    """
    Simulates the entire microservices mesh running locally.
    Routes requests between components.
    """

    def __init__(self):
        print("Initializing System Orchestrator...")

        # 1. User State Encoder Service
        self.encoder = UserStateEncoder()

        # 2. DPO / Preference Service
        self.preference_gen = PreferenceDataGenerator()

        # 3. Monitoring Service
        self.metrics_collector = MetricsCollector()
        self.alert_engine = AlertEngine(
            [
                AlertConfig(
                    name="HighLatency",
                    expr="encoder_latency > 0.5",
                    severity="warning",
                    period_minutes=1,
                    description="Latency High",
                )
            ]
        )

        # 4. Privacy Service
        self.privacy_tracker = PrivacyBudgetTracker()

        print("All services initialized.")

    async def process_user_event(self, user_id: str, event_text: str):
        """
        Simulates end-to-end flow:
        1. Event -> Encoder
        2. Check Privacy
        3. Monitoring tap
        """
        # Privacy Check
        if not self.privacy_tracker.check_budget():
            raise Exception("Privacy Budget Exceeded")

        # 1. Encode
        state_embedding = self.encoder.encode_user_state([event_text])

        # 2. Monitor
        self.metrics_collector.poll_all()  # Simulate periodic poll

        return state_embedding

    async def submit_feedback(
        self, user_id: str, prompt: str, chosen: str, rejected: str
    ):
        """
        Simulate feedback loop
        """
        # 1. Log preference
        self.preference_gen.add_interaction(user_id, prompt, chosen, rejected)

        # 2. Update Privacy Budget (Mock cost)
        self.privacy_tracker.step(noise_multiplier=1.0, sample_rate=0.01)

        return {"status": "accepted"}

    def health_check(self):
        return {
            "encoder": "healthy",
            "dpo": "healthy",
            "monitoring": "healthy",
            "privacy": "healthy",
        }
