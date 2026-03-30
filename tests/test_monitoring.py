"""Tests for Monitoring & Observability."""
from __future__ import annotations
import time
from datetime import datetime
import pytest
from lakehouse_weather.monitoring import (
    EventRecord, HealthChecker, HealthStatus,
    MetricsCollector, PipelineEvent, TaskTimer,
)


class TestMetricsCollector:
    def test_record_event(self):
        mc = MetricsCollector()
        mc.record(PipelineEvent.GENERATION_START, {"rows": 100})
        assert len(mc.events) == 1
        assert mc.events[0].event == PipelineEvent.GENERATION_START
        assert mc.events[0].payload == {"rows": 100}

    def test_counter_incremented(self):
        mc = MetricsCollector()
        mc.record(PipelineEvent.GENERATION_START)
        mc.record(PipelineEvent.GENERATION_START)
        assert mc.counters["event_generation_start"] == 2

    def test_set_gauge(self):
        mc = MetricsCollector()
        mc.set_gauge("latency_ms", 42.5)
        assert mc.gauges["latency_ms"] == 42.5

    def test_increment_counter(self):
        mc = MetricsCollector()
        mc.increment("custom_counter", 3)
        mc.increment("custom_counter", 2)
        assert mc.counters["custom_counter"] == 5

    def test_last_event(self):
        mc = MetricsCollector()
        mc.record(PipelineEvent.QUALITY_CHECK_PASSED, {"score": 0.9})
        mc.record(PipelineEvent.GENERATION_COMPLETE)
        mc.record(PipelineEvent.QUALITY_CHECK_PASSED, {"score": 1.0})
        last = mc.last_event(PipelineEvent.QUALITY_CHECK_PASSED)
        assert last is not None
        assert last.payload["score"] == 1.0

    def test_last_event_none_when_absent(self):
        mc = MetricsCollector()
        mc.record(PipelineEvent.GENERATION_START)
        assert mc.last_event(PipelineEvent.PIPELINE_ERROR) is None

    def test_summary(self):
        mc = MetricsCollector()
        mc.record(PipelineEvent.GENERATION_START)
        mc.set_gauge("rows", 1000)
        s = mc.summary()
        assert s["total_events"] == 1
        assert "rows" in s["gauges"]
        assert s["latest_ts"] is not None

    def test_summary_empty(self):
        mc = MetricsCollector()
        s = mc.summary()
        assert s["total_events"] == 0
        assert s["latest_ts"] is None

    def test_event_hook(self):
        mc = MetricsCollector()
        captured = []
        mc.register_hook(lambda rec: captured.append(rec))
        mc.record(PipelineEvent.PIPELINE_COMPLETE, {"ok": True})
        assert len(captured) == 1
        assert captured[0].event == PipelineEvent.PIPELINE_COMPLETE

    def test_hook_exception_does_not_break(self):
        mc = MetricsCollector()
        mc.register_hook(lambda rec: 1 / 0)
        mc.record(PipelineEvent.GENERATION_START)
        assert len(mc.events) == 1


class TestHealthChecker:
    def test_healthy_when_no_checks(self):
        hc = HealthChecker()
        assert hc.is_healthy()

    def test_custom_check_passes(self):
        hc = HealthChecker()
        hc.register(lambda: HealthStatus("disk", True, "ok"))
        assert hc.is_healthy()

    def test_custom_check_fails(self):
        hc = HealthChecker()
        hc.register(lambda: HealthStatus("disk", False, "full"))
        assert not hc.is_healthy()

    def test_run_all_returns_list(self):
        hc = HealthChecker()
        hc.register(lambda: HealthStatus("a", True, "ok"))
        hc.register(lambda: HealthStatus("b", False, "bad"))
        results = hc.run_all()
        assert len(results) == 2

    def test_check_that_raises_is_marked_unhealthy(self):
        hc = HealthChecker()
        def bad_check():
            raise ValueError("boom")
        hc.register(bad_check)
        results = hc.run_all()
        assert len(results) == 1
        assert not results[0].healthy
        assert "boom" in results[0].message

    def test_metrics_collector_check_healthy(self):
        mc = MetricsCollector()
        mc.record(PipelineEvent.PIPELINE_COMPLETE)
        hc = HealthChecker(metrics=mc)
        status = hc.check_metrics_collector()
        assert status.healthy

    def test_metrics_collector_check_unhealthy(self):
        mc = MetricsCollector()
        mc.record(PipelineEvent.PIPELINE_ERROR, {"error": "test"})
        hc = HealthChecker(metrics=mc)
        status = hc.check_metrics_collector()
        assert not status.healthy

    def test_quality_gate_check_pass(self):
        mc = MetricsCollector()
        mc.record(PipelineEvent.QUALITY_CHECK_PASSED)
        hc = HealthChecker(metrics=mc)
        status = hc.check_quality_gate()
        assert status.healthy

    def test_quality_gate_check_fail(self):
        mc = MetricsCollector()
        mc.record(PipelineEvent.QUALITY_CHECK_PASSED)
        mc.record(PipelineEvent.QUALITY_CHECK_FAILED)
        hc = HealthChecker(metrics=mc)
        status = hc.check_quality_gate()
        assert not status.healthy

    def test_no_metrics_attached(self):
        hc = HealthChecker(metrics=None)
        status = hc.check_metrics_collector()
        assert not status.healthy


class TestTaskTimer:
    def test_timer_records_duration(self):
        mc = MetricsCollector()
        with TaskTimer(mc, "test_task"):
            time.sleep(0.05)
        gauge = mc.gauges.get("task_duration_test_task")
        assert gauge is not None
        assert gauge >= 0.04

    def test_timer_records_start_and_complete(self):
        mc = MetricsCollector()
        with TaskTimer(mc, "my_task"):
            pass
        names = [e.event for e in mc.events]
        assert PipelineEvent.DAG_TASK_START in names
        assert PipelineEvent.DAG_TASK_COMPLETE in names

    def test_timer_records_failure_on_exception(self):
        mc = MetricsCollector()
        with pytest.raises(ValueError):
            with TaskTimer(mc, "fail_task"):
                raise ValueError("whoops")
        names = [e.event for e in mc.events]
        assert PipelineEvent.DAG_TASK_FAILED in names

    def test_timer_with_no_collector(self):
        with TaskTimer(None, "orphan_task"):
            pass


class TestEventRecord:
    def test_fields(self):
        rec = EventRecord(
            event=PipelineEvent.GENERATION_START,
            timestamp=datetime(2024, 1, 1),
            payload={"k": "v"},
        )
        assert rec.event == PipelineEvent.GENERATION_START
        assert rec.payload["k"] == "v"
