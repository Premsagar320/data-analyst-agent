"""Run logger for the data-analyst agent.

Appends one row to logs/runs.csv after every agent run with:
    timestamp, model, question, retries, seconds, had_error
"""

import csv
import os
import time
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "runs.csv")
FIELDNAMES = ["timestamp", "model", "question", "retries", "seconds", "had_error"]


def _ensure_log_file():
    """Create logs/ directory and write CSV header if the file doesn't exist yet."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDNAMES).writeheader()


def log_run(*, model: str, question: str, retries: int, seconds: float, had_error: bool):
    """Append one row to logs/runs.csv."""
    _ensure_log_file()
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "question": question,
        "retries": retries,
        "seconds": round(seconds, 3),
        "had_error": int(had_error),
    }
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=FIELDNAMES).writerow(row)
