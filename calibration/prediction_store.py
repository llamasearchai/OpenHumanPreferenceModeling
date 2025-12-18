import os
import random
import sqlite3
import threading
import time
from typing import Iterable, List, Tuple


class PredictionStoreError(Exception):
    pass


class PredictionStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    confidence REAL NOT NULL,
                    correct INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at)"
            )

    def record(self, confidence: float, correct: int, sample_rate: float) -> bool:
        if confidence < 0.0 or confidence > 1.0:
            raise PredictionStoreError("confidence out of range")
        if correct not in (0, 1):
            raise PredictionStoreError("correct must be 0 or 1")
        if not 0.0 <= sample_rate <= 1.0:
            raise PredictionStoreError("sample_rate out of range")
        if random.random() > sample_rate:
            return False
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO predictions (confidence, correct, created_at) VALUES (?, ?, ?)",
                (confidence, correct, time.time()),
            )
        return True

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
        return int(row[0])

    def fetch(self, limit: int) -> Tuple[List[int], List[float], List[int]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, confidence, correct FROM predictions ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        ids = [int(row[0]) for row in rows]
        confidences = [float(row[1]) for row in rows]
        corrects = [int(row[2]) for row in rows]
        return ids, confidences, corrects

    def delete(self, ids: Iterable[int]) -> None:
        ids_list = list(ids)
        if not ids_list:
            return
        placeholders = ",".join("?" for _ in ids_list)
        with self._lock, self._connect() as conn:
            conn.execute(f"DELETE FROM predictions WHERE id IN ({placeholders})", ids_list)

    def consume(self, limit: int) -> Tuple[List[float], List[int]]:
        ids, confidences, corrects = self.fetch(limit)
        self.delete(ids)
        return confidences, corrects
