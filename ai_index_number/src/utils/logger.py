from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.helpers import read_json, write_json


class JsonLogger:
    def __init__(self, log_path: str = "outputs/logs.json") -> None:
        self.log_path = Path(log_path)

    def log_query(self, payload: dict[str, Any]) -> None:
        logs = read_json(self.log_path, default=[])
        payload["timestamp"] = datetime.utcnow().isoformat()
        logs.append(payload)
        write_json(self.log_path, logs)
