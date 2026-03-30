"""
Pipeline Monitoring & Observability
=====================================
Lightweight, extensible event collector and health-check reporter.
Integrates with structured logging and can push to external sinks
(Prometheus, Azure Monitor, Datadog) via adapters.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline events
# ---------------------------------------------------------------------------
class PipelineEvent(Enum):
    GENERATION_START = auto()
    GENERATION_COMPLETE = auto()
    QUALITY_CHECK_START = auto()
    QUALITY_CHECK_PASSED = auto()
    QUALITY_CHECK_FAILED = auto()
    PIPELINE_COMPLETE = auto()
    PIPELINE_ERROR = auto()
    DAG_TASK_START = auto()
    DAG_TASK_COMPLETE = auto()
    DAG_TASK_FAILED = auto()


@dataclass
class EventRecord:
    event: PipelineEvent
    timestamp: datetime
    payload: Dict[str, Any]


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------
class MetricsCollector:
    """In-process metrics sink.  Records events with timestamps & payloads."""

    def __init__(self):
        self._events: List[EventRecord] = []
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._hooks: List[Callable[[EventRecord], None]] = []

    # --- Recording ---
    def record(self, event: PipelineEvent, payload: Optional[Dict[str, Any]] = None):
        rec = EventRecord(event=event, timestamp=datetime.utcnow(), payload=payload or {})
        self._events.append(rec)
        counter_key = f"event_{event.name.lower()}"
        self._counters[counter_key] = self._counters.get(counter_key, 0) + 1
        logger.info("MONITOR | %s | %s", event.name, json.dumps(payload or {}, default=str))
        for hook in self._hooks:
            try:
                hook(rec)
            except Exception:
                logger.exception("Hook failed for event %s", event.name)

    def set_gauge(self, name: str, value: float):
        self._gauges[name] = value

    def increment(self, name: str, delta: int = 1):
        self._counters[name] = self._counters.get(name, 0) + delta

    def register_hook(self, fn: Callable[[EventRecord], None]):
        """Add external callback invoked on every record()."""
        self._hooks.append(fn)

    # --- Querying ---
    @property
    def events(self) -> List[EventRecord]:
        return list(self._events)

    @property
    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

    @property
    def gauges(self) -> Dict[str, float]:
        return dict(self._gauges)

    def last_event(self, event_type: PipelineEvent) -> Optional[EventRecord]:
        for rec in reversed(self._events):
            if rec.event == event_type:
                return rec
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "total_events": len(self._events),
            "counters": self.counters,
            "gauges": self.gauges,
            "latest_ts": self._events[-1].timestamp.isoformat() if self._events else None,
        }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@dataclass
class HealthStatus:
    component: str
    healthy: bool
    message: str
    checked_at: datetime = field(default_factory=datetime.utcnow)


class HealthChecker:
    """Run a battery of lightweight health checks and return aggregate status."""

    def __init__(self, metrics: Optional[MetricsCollector] = None):
        self.metrics = metrics
        self._checks: List[Callable[[], HealthStatus]] = []

    def register(self, check_fn: Callable[[], HealthStatus]):
        self._checks.append(check_fn)

    def run_all(self) -> List[HealthStatus]:
        results = []
        for fn in self._checks:
            try:
                results.append(fn())
            except Exception as exc:
                results.append(HealthStatus(
                    component=fn.__name__,
                    healthy=False,
                    message=f"check raised: {exc}",
                ))
        return results

    def is_healthy(self) -> bool:
        return all(s.healthy for s in self.run_all())

    # Built-in checks  ------------------------------------------------
    def check_metrics_collector(self) -> HealthStatus:
        if self.metrics is None:
            return HealthStatus("metrics_collector", False, "no collector attached")
        error_count = self.metrics.counters.get("event_pipeline_error", 0)
        ok = error_count == 0
        return HealthStatus(
            "metrics_collector", ok,
            f"pipeline_error count = {error_count}",
        )

    def check_quality_gate(self) -> HealthStatus:
        if self.metrics is None:
            return HealthStatus("quality_gate", False, "no collector attached")
        last_fail = self.metrics.last_event(PipelineEvent.QUALITY_CHECK_FAILED)
        last_pass = self.metrics.last_event(PipelineEvent.QUALITY_CHECK_PASSED)
        if last_fail and (not last_pass or last_fail.timestamp > last_pass.timestamp):
            return HealthStatus("quality_gate", False, "latest quality check FAILED")
        return HealthStatus("quality_gate", True, "quality gate passed or not yet run")


# ---------------------------------------------------------------------------
# Timer context manager (for DAG task instrumentation)
# ---------------------------------------------------------------------------
class TaskTimer:
    """Usage:  with TaskTimer(metrics, 'my_task') as t: ..."""

    def __init__(self, collector: Optional[MetricsCollector], task_name: str):
        self.collector = collector
        self.task_name = task_name
        self._start: float = 0

    def __enter__(self):
        self._start = time.monotonic()
        if self.collector:
            self.collector.record(PipelineEvent.DAG_TASK_START, {"task": self.task_name})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.monotonic() - self._start
        if self.collector:
            self.collector.set_gauge(f"task_duration_{self.task_name}", elapsed)
            event = PipelineEvent.DAG_TASK_COMPLETE if exc_type is None else PipelineEvent.DAG_TASK_FAILED
            self.collector.record(event, {"task": self.task_name, "duration_s": round(elapsed, 3)})
        return False  # don't swallow exceptions
