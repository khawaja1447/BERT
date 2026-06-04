import json
import logging
import os
import time
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def batch_texts(texts: List[str], batch_size: int) -> List[List[str]]:
    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]


class LatencyTracker:
    """Rolling window latency stats for monitoring."""

    def __init__(self, window: int = 1000):
        self._window = window
        self._samples: List[float] = []

    def record(self, ms: float) -> None:
        self._samples.append(ms)
        if len(self._samples) > self._window:
            self._samples.pop(0)

    def stats(self) -> Dict[str, float]:
        if not self._samples:
            return {}
        arr = np.array(self._samples)
        return {
            "count": len(arr),
            "mean_ms": round(float(arr.mean()), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
        }


class PredictionCounter:
    """Thread-safe simple counter."""

    def __init__(self):
        self._count = 0
        self._start = time.time()

    def increment(self, n: int = 1) -> None:
        self._count += n

    @property
    def total(self) -> int:
        return self._count

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self._start, 1)
