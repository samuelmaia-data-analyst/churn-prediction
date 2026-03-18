from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def __init__(self, run_id: str, environment: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": self.run_id,
            "environment": self.environment,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str, log_dir: Path, run_id: str, environment: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{run_id}.log"
    latest_log = log_dir / "pipeline.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    formatter = JsonFormatter(run_id=run_id, environment=environment)

    stream_handler = logging.StreamHandler(stream=sys.__stdout__)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    latest_handler = logging.FileHandler(latest_log, encoding="utf-8")
    latest_handler.setFormatter(formatter)
    root.addHandler(latest_handler)
