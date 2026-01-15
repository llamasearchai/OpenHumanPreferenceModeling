import time
from typing import Dict, Any


class AuditLog:
    """
    Simulates immutable append-only log.
    """

    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
        self.entries = []

    def log_access(self, accessor_id: str, subject_id: str, action: str, purpose: str):
        entry = {
            "timestamp": time.time(),
            "accessor_id": accessor_id,
            "subject_id": subject_id,
            "action": action,
            "purpose": purpose,
        }
        self.entries.append(entry)
        # In real life, write to QLDB or signed file
        # print(f"AUDIT LOG: {entry}")

    def get_logs(self, subject_id: str):
        return [e for e in self.entries if e["subject_id"] == subject_id]
