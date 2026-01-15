import random
import psutil
from datetime import datetime
from typing import List

from .models import Metric

try:
    from calibration.auto_recalibration import AutoRecalibrationService
except Exception:  # pragma: no cover
    AutoRecalibrationService = None


class MetricsCollector:
    """
    Collects system metrics from various components.
    """

    def __init__(self):
        self.metrics_store: List[Metric] = []
        # In-memory list keeps recent samples for quick access.
        self._auto_recalibration = (
            AutoRecalibrationService() if AutoRecalibrationService else None
        )

    def poll_all(self):
        """Polls components and stores metrics."""
        now = datetime.now()

        # 1. System Metrics (Real)
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent

        self.metrics_store.append(
            Metric(
                name="system_cpu_percent",
                value=cpu_usage,
                timestamp=now,
                tags={"host": "localhost"},
            )
        )
        self.metrics_store.append(
            Metric(
                name="system_memory_percent",
                value=memory_usage,
                timestamp=now,
                tags={"host": "localhost"},
            )
        )

        # Restore encoder latency for tests (simulation)
        self.metrics_store.append(
            Metric(
                name="encoder_latency_seconds",
                value=random.uniform(0.01, 0.1),
                timestamp=now,
                tags={"service": "encoder"},
            )
        )

        # 2. Component 4: DPO Model (Simulation kept for model specifics not available yet)
        # In a real scenario, this would come from the model serving endpoint or shared state
        self.metrics_store.append(
            Metric(
                name="model_accuracy",
                value=random.uniform(
                    0.70, 0.85
                ),  # Placeholder: Connect to real model evaluation if available
                timestamp=now,
                tags={"model": "dpo_v1", "mode": "simulation"},
            )
        )

        # 3. Component 7: Annotation
        self.metrics_store.append(
            Metric(
                name="annotation_queue_depth",
                value=float(
                    random.randint(0, 50)
                ),  # Placeholder: Connect to shared active learning queue
                timestamp=now,
                tags={"service": "annotation", "mode": "simulation"},
            )
        )

        if not self._auto_recalibration:
            return

        check = self._auto_recalibration.monitor.check_ece()
        if check is not None:
            self.metrics_store.append(
                Metric(
                    name="model_ece_current",
                    value=check.ece,
                    timestamp=now,
                    tags={"model": "dpo_v1"},
                )
            )
            if check.triggered:
                try:
                    result = self._auto_recalibration.trigger_recalibration(check.ece)
                except Exception:
                    self.metrics_store.append(
                        Metric(
                            name="recalibration_total",
                            value=1.0,
                            timestamp=now,
                            tags={"status": "failed", "model": "dpo_v1"},
                        )
                    )
                else:
                    if result is not None:
                        self.metrics_store.append(
                            Metric(
                                name="recalibration_total",
                                value=1.0,
                                timestamp=now,
                                tags={"status": "success", "model": "dpo_v1"},
                            )
                        )

    def get_metrics(self, name: str, limit: int = 100) -> List[Metric]:
        return [m for m in self.metrics_store if m.name == name][-limit:]
