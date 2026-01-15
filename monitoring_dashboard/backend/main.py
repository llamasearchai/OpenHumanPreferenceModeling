from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from typing import List
import yaml
import asyncio
from datetime import datetime

from .models import Metric, Alert, AlertConfig
from .metrics_collector import MetricsCollector
from .alert_engine import AlertEngine

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(run_poller())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Monitoring Dashboard API", lifespan=lifespan)

# Initialize
collector = MetricsCollector()
alert_configs = []

try:
    with open("configs/monitoring_config.yaml", "r") as f:
        conf = yaml.safe_load(f)
        for r in conf.get("rules", []):
            alert_configs.append(
                AlertConfig(
                    name=r["name"],
                    expr=r["expr"],
                    severity=r["severity"],
                    period_minutes=r["period_minutes"],
                    description=r["description"],
                )
            )
except Exception as e:
    print(f"Failed to load config: {e}")

alert_engine = AlertEngine(alert_configs)


async def run_poller():
    while True:
        collector.poll_all()
        # Feed all metrics to alert engine
        # In real life, we'd query window.
        all_metrics = collector.metrics_store
        alert_engine.evaluate(all_metrics)
        await asyncio.sleep(5)  # Poll every 5s for demo


@app.get("/api/metrics", response_model=List[Metric])
async def get_metrics(name: str):
    return collector.get_metrics(name)


@app.get("/api/alerts", response_model=List[Alert])
async def get_alerts():
    return alert_engine.get_alerts()


@app.post("/api/alerts/{alert_id}/ack")
async def ack_alert(alert_id: str):
    alert_engine.ack_alert(alert_id)
    return {"status": "success"}
