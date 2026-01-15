from typing import List, Dict
from datetime import datetime
import uuid
from .models import Alert, AlertConfig, Metric


class AlertEngine:
    def __init__(self, configs: List[AlertConfig]):
        self.configs = configs
        self.active_alerts: Dict[str, Alert] = {}  # Keyed by rule_name

    def evaluate(self, metrics: List[Metric]):
        """
        Evaluate rules against latest metrics.
        Very simple logic: just check if ANY recent metric violates threshold.
        Real implementation would use sliding window averages.
        """
        for config in self.configs:
            # Parse simple expr like "metric > val"
            try:
                lhs, op, rhs = config.expr.split()
                rhs_val = float(rhs)

                # Find latest metric matching LHS
                relevant_metrics = [m for m in metrics if m.name == lhs]
                if not relevant_metrics:
                    continue

                latest = relevant_metrics[-1]

                is_firing = False
                if op == ">":
                    is_firing = latest.value > rhs_val
                elif op == "<":
                    is_firing = latest.value < rhs_val

                if is_firing:
                    if config.name not in self.active_alerts:
                        # Trigger new alert
                        alert_id = str(uuid.uuid4())
                        self.active_alerts[config.name] = Alert(
                            id=alert_id,
                            rule_name=config.name,
                            severity=config.severity,
                            status="firing",
                            timestamp=datetime.now(),
                            message=f"{config.description} (Value: {latest.value:.2f})",
                        )
                else:
                    # Auto-resolve
                    if config.name in self.active_alerts:
                        del self.active_alerts[config.name]

            except Exception as e:
                print(f"Error evaluating rule {config.name}: {e}")

    def get_alerts(self) -> List[Alert]:
        return list(self.active_alerts.values())

    def ack_alert(self, alert_id: str):
        for name, alert in self.active_alerts.items():
            if alert.id == alert_id:
                alert.status = "acknowledged"
