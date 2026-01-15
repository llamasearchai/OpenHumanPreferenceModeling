import io
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from scipy.stats import ks_2samp
from tenacity import retry, stop_after_attempt, wait_exponential

from calibration.metrics import CalibrationMetrics
from calibration.prediction_store import PredictionStore, PredictionStoreError
from calibration.recalibration import TemperatureScaler


logger = logging.getLogger(__name__)


class RecalibrationError(Exception):
    """Base exception for all recalibration errors."""


class InsufficientDataError(RecalibrationError):
    """Raised when there is not enough data for recalibration."""


class ValidationDataError(RecalibrationError):
    """Raised when validation data cannot be loaded or is invalid."""


class ConvergenceFailedError(RecalibrationError):
    """Raised when temperature optimization fails to converge."""


class RecalibrationInProgressError(RecalibrationError):
    """Raised when a recalibration job is already running."""


@dataclass
class CalibrationSettings:
    sample_rate: float
    ece_threshold: float
    check_interval_seconds: int
    consecutive_checks: int
    min_samples: int
    max_iterations: int
    temperature_bounds: Tuple[float, float]
    validation_data_uri: Optional[str]
    ks_reference_uri: Optional[str]
    ks_alpha: float
    kafka_topic: str
    kafka_servers: Optional[str]
    slack_webhook: Optional[str]
    model_version: Optional[str]
    prediction_store_path: str
    configmap_name: Optional[str]
    configmap_namespace: Optional[str]
    configmap_key: Optional[str]
    restart_kind: str
    restart_name: Optional[str]
    rollout_timeout_seconds: int


@dataclass
class RecalibrationResult:
    temperature: float
    pre_ece: float
    post_ece: float
    iterations: int
    warning: Optional[str] = None


@dataclass
class EceCheckResult:
    ece: float
    triggered: bool


def load_calibration_settings() -> CalibrationSettings:
    config_path = os.getenv("CALIBRATION_CONFIG", "configs/calibration_config.yaml")
    config: Dict[str, Dict[str, float]] = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as handle:
            config = yaml.safe_load(handle) or {}

    calibration_cfg = config.get("calibration", {})

    def _get_env_float(name: str, fallback: float) -> float:
        value = os.getenv(name)
        if value is None:
            return fallback
        return float(value)

    def _get_env_int(name: str, fallback: int) -> int:
        value = os.getenv(name)
        if value is None:
            return fallback
        return int(value)

    sample_rate = _get_env_float(
        "CALIBRATION_SAMPLE_RATE", calibration_cfg.get("sample_rate", 0.01)
    )
    ece_threshold = _get_env_float(
        "CALIBRATION_ECE_THRESHOLD", calibration_cfg.get("ece_threshold", 0.15)
    )
    check_interval_seconds = _get_env_int(
        "CALIBRATION_CHECK_INTERVAL_SECONDS",
        calibration_cfg.get("check_interval_seconds", 3600),
    )
    consecutive_checks = _get_env_int(
        "CALIBRATION_CONSECUTIVE_CHECKS", calibration_cfg.get("consecutive_checks", 3)
    )
    min_samples = _get_env_int(
        "CALIBRATION_MIN_SAMPLES", calibration_cfg.get("min_samples", 1000)
    )
    max_iterations = _get_env_int(
        "CALIBRATION_MAX_ITERATIONS", calibration_cfg.get("max_iterations", 100)
    )
    temp_min = _get_env_float(
        "CALIBRATION_TEMPERATURE_MIN", calibration_cfg.get("temperature_min", 0.5)
    )
    temp_max = _get_env_float(
        "CALIBRATION_TEMPERATURE_MAX", calibration_cfg.get("temperature_max", 5.0)
    )
    validation_data_uri = os.getenv(
        "CALIBRATION_VALIDATION_DATA_URI", calibration_cfg.get("validation_data_uri")
    )
    ks_reference_uri = os.getenv(
        "CALIBRATION_REFERENCE_DATA_URI", calibration_cfg.get("reference_data_uri")
    )
    ks_alpha = _get_env_float(
        "CALIBRATION_POISONING_ALPHA", calibration_cfg.get("poisoning_alpha", 0.05)
    )
    kafka_topic = os.getenv(
        "CALIBRATION_KAFKA_TOPIC",
        calibration_cfg.get("kafka_topic", "calibration-events"),
    )
    kafka_servers = os.getenv(
        "CALIBRATION_KAFKA_SERVERS", calibration_cfg.get("kafka_servers")
    )
    slack_webhook = os.getenv(
        "CALIBRATION_SLACK_WEBHOOK", calibration_cfg.get("slack_webhook")
    )
    model_version = os.getenv(
        "CALIBRATION_MODEL_VERSION", calibration_cfg.get("model_version")
    )
    prediction_store_path = os.getenv(
        "CALIBRATION_PREDICTION_STORE_PATH",
        calibration_cfg.get("prediction_store_path", "calibration_predictions.sqlite"),
    )
    configmap_name = os.getenv(
        "CALIBRATION_CONFIGMAP_NAME", calibration_cfg.get("configmap_name")
    )
    configmap_namespace = os.getenv(
        "CALIBRATION_CONFIGMAP_NAMESPACE", calibration_cfg.get("configmap_namespace")
    )
    configmap_key = os.getenv(
        "CALIBRATION_CONFIGMAP_KEY", calibration_cfg.get("configmap_key")
    )
    restart_kind = os.getenv(
        "CALIBRATION_RESTART_KIND", calibration_cfg.get("restart_kind", "deployment")
    )
    restart_name = os.getenv(
        "CALIBRATION_RESTART_NAME", calibration_cfg.get("restart_name")
    )
    rollout_timeout_seconds = _get_env_int(
        "CALIBRATION_ROLLOUT_TIMEOUT_SECONDS",
        calibration_cfg.get("rollout_timeout_seconds", 300),
    )

    return CalibrationSettings(
        sample_rate=sample_rate,
        ece_threshold=ece_threshold,
        check_interval_seconds=check_interval_seconds,
        consecutive_checks=consecutive_checks,
        min_samples=min_samples,
        max_iterations=max_iterations,
        temperature_bounds=(temp_min, temp_max),
        validation_data_uri=validation_data_uri,
        ks_reference_uri=ks_reference_uri,
        ks_alpha=ks_alpha,
        kafka_topic=kafka_topic,
        kafka_servers=kafka_servers,
        slack_webhook=slack_webhook,
        model_version=model_version,
        prediction_store_path=prediction_store_path,
        configmap_name=configmap_name,
        configmap_namespace=configmap_namespace,
        configmap_key=configmap_key,
        restart_kind=restart_kind,
        restart_name=restart_name,
        rollout_timeout_seconds=rollout_timeout_seconds,
    )


def _log_event(event: str, payload: Dict[str, object]) -> None:
    logger.info(json.dumps({"event": event, **payload}))


class EventPublisher:
    def publish(self, event: str, payload: Dict[str, object]) -> None:
        raise NotImplementedError


class KafkaEventPublisher(EventPublisher):
    def __init__(self, servers: str, topic: str):
        self.topic = topic
        try:
            from kafka import KafkaProducer
        except ImportError as exc:
            raise RecalibrationError("kafka-python not installed") from exc
        self.producer = KafkaProducer(bootstrap_servers=servers)

    def publish(self, event: str, payload: Dict[str, object]) -> None:
        message = json.dumps({"event": event, **payload}).encode("utf-8")
        self.producer.send(self.topic, message)
        self.producer.flush(timeout=5)


class Notifier:
    def notify(self, message: str) -> None:
        raise NotImplementedError


class SlackNotifier(Notifier):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def notify(self, message: str) -> None:
        import urllib.request

        payload = json.dumps({"text": message}).encode("utf-8")
        request = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=10)


class EceThresholdTracker:
    def __init__(self, threshold: float, consecutive_required: int):
        self.threshold = threshold
        self.consecutive_required = consecutive_required
        self._consecutive = 0

    def record(self, ece_value: float) -> bool:
        if ece_value > self.threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self._consecutive >= self.consecutive_required


class CalibrationMonitor:
    def __init__(
        self,
        settings: CalibrationSettings,
        store: Optional[PredictionStore] = None,
        n_bins: int = 10,
    ):
        self.settings = settings
        self.store = store or PredictionStore(settings.prediction_store_path)
        self._tracker = EceThresholdTracker(
            threshold=settings.ece_threshold,
            consecutive_required=settings.consecutive_checks,
        )
        self._metrics = CalibrationMetrics(n_bins=n_bins)

    def record_prediction(self, confidence: float, correct: int) -> bool:
        try:
            return self.store.record(confidence, correct, self.settings.sample_rate)
        except PredictionStoreError as exc:
            raise ValidationDataError(str(exc)) from exc

    def check_ece(self) -> Optional[EceCheckResult]:
        available = self.store.count()
        if available < self.settings.min_samples:
            return None
        confidences, corrects = self.store.consume(available)
        preds = np.array(confidences)
        labels = np.array(corrects)
        ece_value = float(self._metrics.compute_ece(preds, labels))
        triggered = self._tracker.record(ece_value)
        return EceCheckResult(ece=ece_value, triggered=triggered)

    def record_ece_value(self, ece_value: float) -> bool:
        return self._tracker.record(ece_value)


class LocalLock:
    def __init__(self):
        self._locked = False

    def acquire(self) -> bool:
        if self._locked:
            return False
        self._locked = True
        return True

    def release(self) -> None:
        self._locked = False


class RedisLock:
    def __init__(self, client, key: str, ttl_seconds: int = 900):
        self.client = client
        self.key = key
        self.ttl_seconds = ttl_seconds
        self._token = os.urandom(16).hex()

    def acquire(self) -> bool:
        return bool(
            self.client.set(self.key, self._token, nx=True, ex=self.ttl_seconds)
        )

    def release(self) -> None:
        if self.client.get(self.key) == self._token.encode("utf-8"):
            self.client.delete(self.key)


def create_lock() -> LocalLock | RedisLock:
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return LocalLock()
    try:
        import redis
    except ImportError:
        return LocalLock()
    client = redis.Redis.from_url(redis_url)
    return RedisLock(client, key="calibration:recalibration_lock")


def build_event_publisher(settings: CalibrationSettings) -> EventPublisher:
    if not settings.kafka_servers:
        raise RecalibrationError("CALIBRATION_KAFKA_SERVERS is required")
    return KafkaEventPublisher(
        servers=settings.kafka_servers, topic=settings.kafka_topic
    )


def build_notifier(settings: CalibrationSettings) -> Notifier:
    if not settings.slack_webhook:
        raise RecalibrationError("CALIBRATION_SLACK_WEBHOOK is required")
    return SlackNotifier(settings.slack_webhook)


def build_temperature_applier(settings: CalibrationSettings) -> "TemperatureApplier":
    missing = []
    if not settings.configmap_name:
        missing.append("CALIBRATION_CONFIGMAP_NAME")
    if not settings.configmap_namespace:
        missing.append("CALIBRATION_CONFIGMAP_NAMESPACE")
    if not settings.configmap_key:
        missing.append("CALIBRATION_CONFIGMAP_KEY")
    if not settings.restart_name:
        missing.append("CALIBRATION_RESTART_NAME")
    if missing:
        raise RecalibrationError(f"Missing config: {', '.join(missing)}")
    return KubernetesTemperatureApplier(
        namespace=settings.configmap_namespace,
        configmap_name=settings.configmap_name,
        configmap_key=settings.configmap_key,
        restart_kind=settings.restart_kind,
        restart_name=settings.restart_name,
        rollout_timeout_seconds=settings.rollout_timeout_seconds,
    )


class TemperatureApplier:
    def apply(self, temperature: float) -> None:
        raise NotImplementedError


class KubernetesTemperatureApplier(TemperatureApplier):
    def __init__(
        self,
        namespace: str,
        configmap_name: str,
        configmap_key: str,
        restart_kind: str,
        restart_name: str,
        rollout_timeout_seconds: int = 300,
    ):
        self.namespace = namespace
        self.configmap_name = configmap_name
        self.configmap_key = configmap_key
        self.restart_kind = restart_kind
        self.restart_name = restart_name
        self.rollout_timeout_seconds = rollout_timeout_seconds

    def apply(self, temperature: float) -> None:
        patch = {"data": {self.configmap_key: f"{temperature:.6f}"}}
        self._kubectl(
            [
                "patch",
                "configmap",
                self.configmap_name,
                "--type",
                "merge",
                "-p",
                json.dumps(patch),
            ]
        )
        restart_target = f"{self.restart_kind}/{self.restart_name}"
        self._kubectl(["rollout", "restart", restart_target])
        self._kubectl(
            [
                "rollout",
                "status",
                restart_target,
                f"--timeout={self.rollout_timeout_seconds}s",
            ]
        )

    def _kubectl(self, args: list[str]) -> None:
        import subprocess

        command = ["kubectl", "-n", self.namespace, *args]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RecalibrationError(result.stderr.strip() or "kubectl command failed")


class ValidationDataFetcher:
    def __init__(self):
        self._local_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def fetch(self, uri: str) -> Tuple[np.ndarray, np.ndarray]:
        if uri.startswith("s3://"):
            return self._fetch_s3(uri)
        return self._fetch_local(uri)

    def _fetch_local(self, uri: str) -> Tuple[np.ndarray, np.ndarray]:
        path = uri.replace("file://", "")
        if not os.path.exists(path):
            raise ValidationDataError("Validation data not found")
        if path in self._local_cache:
            return self._local_cache[path]
        data = np.load(path)
        logits, labels = _extract_logits_labels(data)
        self._local_cache[path] = (logits, labels)
        return logits, labels

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    def _fetch_s3(self, uri: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import boto3
        except ImportError as exc:
            raise ValidationDataError("boto3 not available for s3 access") from exc
        bucket, key = _split_s3_uri(uri)
        client = boto3.client("s3")
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        data = np.load(io.BytesIO(body))
        return _extract_logits_labels(data)


def _split_s3_uri(uri: str) -> Tuple[str, str]:
    stripped = uri.replace("s3://", "")
    parts = stripped.split("/", 1)
    if len(parts) != 2:
        raise ValidationDataError("Invalid S3 URI")
    return parts[0], parts[1]


def _extract_logits_labels(
    data: np.lib.npyio.NpzFile | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(data, np.lib.npyio.NpzFile):
        if "logits" not in data or "labels" not in data:
            raise ValidationDataError("Validation data missing logits or labels")
        logits = np.array(data["logits"])
        labels = np.array(data["labels"])
        data.close()
    else:
        raise ValidationDataError("Unsupported validation data format")

    if logits.shape[0] != labels.shape[0]:
        raise ValidationDataError("Logits/labels length mismatch")
    return logits, labels


def _iter_batches(array: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for idx in range(0, len(array), batch_size):
        yield array[idx : idx + batch_size]


def _prepare_labels(labels: np.ndarray) -> np.ndarray:
    if labels.ndim == 2:
        return labels.argmax(axis=1)
    return labels


def _logits_to_confidence(
    logits: np.ndarray, temperature: float
) -> Tuple[np.ndarray, np.ndarray]:
    scaled = logits / temperature
    if scaled.ndim == 1:
        scaled = np.stack([np.zeros_like(scaled), scaled], axis=1)
    exp_logits = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    pred_class = np.argmax(probs, axis=1)
    confidence = probs[np.arange(len(probs)), pred_class]
    return confidence, pred_class


def compute_ece_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    n_bins: int = 10,
    batch_size: int = 10000,
) -> float:
    metrics = CalibrationMetrics(n_bins=n_bins)
    bin_boundaries = metrics.bin_boundaries
    counts = np.zeros(n_bins, dtype=np.int64)
    acc_sums = np.zeros(n_bins, dtype=np.float64)
    conf_sums = np.zeros(n_bins, dtype=np.float64)

    labels = _prepare_labels(labels)
    for batch_logits, batch_labels in zip(
        _iter_batches(logits, batch_size), _iter_batches(labels, batch_size)
    ):
        confidences, pred_class = _logits_to_confidence(batch_logits, temperature)
        correct = (pred_class == batch_labels).astype(np.float64)
        for idx in range(n_bins):
            if idx == n_bins - 1:
                mask = (confidences >= bin_boundaries[idx]) & (
                    confidences <= bin_boundaries[idx + 1]
                )
            else:
                mask = (confidences >= bin_boundaries[idx]) & (
                    confidences < bin_boundaries[idx + 1]
                )
            if not np.any(mask):
                continue
            counts[idx] += mask.sum()
            acc_sums[idx] += correct[mask].sum()
            conf_sums[idx] += confidences[mask].sum()

    total = counts.sum()
    if total == 0:
        return 0.0

    ece = 0.0
    for idx in range(n_bins):
        if counts[idx] == 0:
            continue
        bin_accuracy = acc_sums[idx] / counts[idx]
        bin_confidence = conf_sums[idx] / counts[idx]
        ece += (counts[idx] / total) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def _optimize_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iterations: int,
    bounds: Tuple[float, float],
    batch_size: int = 10000,
) -> Tuple[float, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = TemperatureScaler().to(device)
    labels = _prepare_labels(labels)
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iterations)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        total_loss = 0.0
        batches = 0
        for batch_logits, batch_labels in zip(
            _iter_batches(logits, batch_size), _iter_batches(labels, batch_size)
        ):
            batch_logits_t = torch.tensor(
                batch_logits, dtype=torch.float32, device=device
            )
            batch_labels_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
            loss = criterion(scaler(batch_logits_t), batch_labels_t)
            loss.backward()
            total_loss += loss.item()
            batches += 1
        avg_loss = total_loss / max(batches, 1)
        return torch.tensor(avg_loss, dtype=torch.float32, device=device)

    optimizer.step(closure)
    temperature = float(scaler.temperature.item())
    temperature = clamp_temperature(temperature, bounds)
    scaler.temperature.data = torch.tensor([temperature], device=device)
    iterations = optimizer.state.get(scaler.temperature, {}).get(
        "n_iter", max_iterations
    )
    return temperature, iterations


def clamp_temperature(value: float, bounds: Tuple[float, float]) -> float:
    return float(np.clip(value, bounds[0], bounds[1]))


def optimize_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iterations: int,
    bounds: Tuple[float, float],
    batch_size: int = 10000,
) -> Tuple[float, int]:
    return _optimize_temperature(
        logits=logits,
        labels=labels,
        max_iterations=max_iterations,
        bounds=bounds,
        batch_size=batch_size,
    )


def _run_ks_test(
    logits: np.ndarray,
    reference_logits: np.ndarray,
    temperature: float,
    alpha: float,
) -> None:
    current_confidence, _ = _logits_to_confidence(logits, temperature)
    reference_confidence, _ = _logits_to_confidence(reference_logits, temperature)
    _, p_value = ks_2samp(reference_confidence, current_confidence)
    if p_value < alpha:
        raise ValidationDataError("Validation data failed KS-test against reference")


class RecalibrationRunner:
    def __init__(
        self,
        settings: CalibrationSettings,
        data_fetcher: Optional[ValidationDataFetcher] = None,
        lock=None,
        event_publisher: Optional[EventPublisher] = None,
        notifier: Optional[Notifier] = None,
        temperature_applier: Optional[TemperatureApplier] = None,
    ):
        self.settings = settings
        self.data_fetcher = data_fetcher or ValidationDataFetcher()
        self.lock = lock or create_lock()
        self.event_publisher = event_publisher
        self.notifier = notifier
        self.temperature_applier = temperature_applier

    def recalibrate(
        self,
        validation_data_uri: str,
        target_ece: float,
        max_iterations: int,
    ) -> RecalibrationResult:
        if not self.lock.acquire():
            raise RecalibrationInProgressError("Recalibration already running")
        start_time = time.time()
        warning = None
        try:
            event_publisher = self.event_publisher or build_event_publisher(
                self.settings
            )
            notifier = self.notifier or build_notifier(self.settings)
            temperature_applier = self.temperature_applier or build_temperature_applier(
                self.settings
            )
            logits, labels = self.data_fetcher.fetch(validation_data_uri)
            if len(labels) < self.settings.min_samples:
                raise InsufficientDataError("Not enough samples for recalibration")

            if self.settings.ks_reference_uri:
                reference_logits, _ = self.data_fetcher.fetch(
                    self.settings.ks_reference_uri
                )
                _run_ks_test(logits, reference_logits, 1.0, self.settings.ks_alpha)

            pre_ece = compute_ece_from_logits(logits, labels, temperature=1.0)
            temperature, iterations = _optimize_temperature(
                logits,
                labels,
                max_iterations=max_iterations,
                bounds=self.settings.temperature_bounds,
            )
            post_ece = compute_ece_from_logits(logits, labels, temperature=temperature)

            if not np.isfinite(post_ece) or not np.isfinite(temperature):
                raise ConvergenceFailedError("Temperature optimization failed")

            if iterations >= max_iterations:
                warning = "max_iterations_reached"
            if post_ece > target_ece:
                warning = (
                    "target_ece_not_met"
                    if warning is None
                    else f"{warning},target_ece_not_met"
                )

            temperature_applier.apply(temperature)
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "temperature": temperature,
                "ece_improvement": pre_ece - post_ece,
                "pre_ece": pre_ece,
                "post_ece": post_ece,
                "target_ece": target_ece,
                "iterations": iterations,
                "duration_seconds": time.time() - start_time,
                "model_version": self.settings.model_version,
                "warning": warning,
            }
            event_publisher.publish("recalibration_completed", payload)
            try:
                notifier.notify(
                    "Recalibration completed: pre_ece={:.4f}, post_ece={:.4f}, temperature={:.3f}".format(
                        pre_ece, post_ece, temperature
                    )
                )
            except Exception as notify_exc:
                _log_event("notification_error", {"error": str(notify_exc)})

            return RecalibrationResult(
                temperature=temperature,
                pre_ece=pre_ece,
                post_ece=post_ece,
                iterations=iterations,
                warning=warning,
            )
        except RecalibrationError as exc:
            try:
                notifier = self.notifier or build_notifier(self.settings)
                notifier.notify(f"Recalibration failed: {exc}")
            except Exception as notify_exc:
                _log_event("notification_error", {"error": str(notify_exc)})
            try:
                event_publisher = self.event_publisher or build_event_publisher(
                    self.settings
                )
                event_publisher.publish(
                    "recalibration_failed",
                    {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "error": str(exc),
                        "model_version": self.settings.model_version,
                    },
                )
            except RecalibrationError as publish_exc:
                _log_event("event_publish_error", {"error": str(publish_exc)})
            raise
        finally:
            self.lock.release()


class AutoRecalibrationService:
    def __init__(self, settings: Optional[CalibrationSettings] = None):
        self.settings = settings or load_calibration_settings()
        self.monitor = CalibrationMonitor(self.settings)
        self.runner = RecalibrationRunner(self.settings)

    def trigger_recalibration(self, ece_value: float) -> Optional[RecalibrationResult]:
        if not self.settings.validation_data_uri:
            raise RecalibrationError("CALIBRATION_VALIDATION_DATA_URI is required")
        _log_event(
            "recalibration_triggered",
            {"ece_value": ece_value, "threshold": self.settings.ece_threshold},
        )
        return self.runner.recalibrate(
            validation_data_uri=self.settings.validation_data_uri,
            target_ece=self.settings.ece_threshold,
            max_iterations=self.settings.max_iterations,
        )

    def observe_ece(self, ece_value: float) -> Optional[RecalibrationResult]:
        triggered = self.monitor.record_ece_value(ece_value)
        if not triggered:
            return None
        return self.trigger_recalibration(ece_value)
