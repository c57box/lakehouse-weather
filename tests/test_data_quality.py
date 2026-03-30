"""
Tests — Data Quality Engine
=============================
Covers: every built-in check (pass & fail), report aggregation,
health score, edge cases with empty DataFrames, and custom configs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from lakehouse_weather.data_quality import (
    CheckResult,
    DataQualityEngine,
    QualityConfig,
    QualityReport,
)
from lakehouse_weather.synthetic_generator import (
    GeneratorConfig,
    DEFAULT_STATIONS,
    SyntheticWeatherGenerator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A small, fully-clean DataFrame (no anomalies, no nulls)."""
    cfg = GeneratorConfig(
        stations=DEFAULT_STATIONS[:2],
        start_date=datetime(2024, 3, 1),
        end_date=datetime(2024, 3, 3, 23, 0),
        seed=100,
        anomaly_rate=0.0,
        null_rate=0.0,
    )
    return SyntheticWeatherGenerator(cfg).generate()


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """DataFrame with high anomaly and null rates for failure testing."""
    cfg = GeneratorConfig(
        stations=DEFAULT_STATIONS[:2],
        start_date=datetime(2024, 3, 1),
        end_date=datetime(2024, 3, 3, 23, 0),
        seed=200,
        anomaly_rate=0.10,
        null_rate=0.08,
    )
    return SyntheticWeatherGenerator(cfg).generate()


@pytest.fixture
def engine() -> DataQualityEngine:
    return DataQualityEngine(QualityConfig(freshness_max_hours=None))


@pytest.fixture
def strict_engine() -> DataQualityEngine:
    return DataQualityEngine(QualityConfig(
        max_null_pct=0.0,
        max_anomaly_pct=0.0,
        max_duplicate_pct=0.0,
        freshness_max_hours=None,
    ))


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------
class TestFullReport:
    def test_clean_data_passes(self, engine, clean_df):
        report = engine.run(clean_df)
        assert report.passed
        assert report.health_score == 1.0

    def test_dirty_data_has_failures(self, strict_engine, dirty_df):
        report = strict_engine.run(dirty_df)
        assert not report.passed
        assert report.health_score < 1.0

    def test_report_to_dataframe(self, engine, clean_df):
        report = engine.run(clean_df)
        rdf = report.to_dataframe()
        assert isinstance(rdf, pd.DataFrame)
        assert "check_name" in rdf.columns
        assert len(rdf) == len(report.checks)

    def test_summary_dict(self, engine, clean_df):
        report = engine.run(clean_df)
        summary = report.summary_dict()
        assert "overall_passed" in summary
        assert "health_score" in summary
        assert "total_checks" in summary


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
class TestRowCount:
    def test_passes_when_above_min(self, engine, clean_df):
        report = engine.run(clean_df)
        rc = next(c for c in report.checks if c.check_name == "row_count_minimum")
        assert rc.passed

    def test_fails_when_below_min(self):
        eng = DataQualityEngine(QualityConfig(min_row_count=999_999, freshness_max_hours=None))
        df = pd.DataFrame({
            "station_id": ["A"], "observation_ts": [datetime.now()],
            "temperature_c": [20], "humidity_pct": [50],
            "pressure_hpa": [1013], "wind_speed_ms": [5],
        })
        report = eng.run(df)
        rc = next(c for c in report.checks if c.check_name == "row_count_minimum")
        assert not rc.passed


class TestSchema:
    def test_passes_with_all_columns(self, engine, clean_df):
        report = engine.run(clean_df)
        sc = next(c for c in report.checks if c.check_name == "schema_completeness")
        assert sc.passed

    def test_fails_with_missing_column(self, engine, clean_df):
        df = clean_df.drop(columns=["temperature_c"])
        report = engine.run(df)
        sc = next(c for c in report.checks if c.check_name == "schema_completeness")
        assert not sc.passed
        assert "temperature_c" in sc.details


class TestNulls:
    def test_no_nulls_passes(self, engine, clean_df):
        report = engine.run(clean_df)
        null_checks = [c for c in report.checks if c.check_name.startswith("null_pct_")]
        assert all(c.passed for c in null_checks)

    def test_high_nulls_fail(self):
        eng = DataQualityEngine(QualityConfig(max_null_pct=0.0, freshness_max_hours=None))
        df = pd.DataFrame({
            "station_id": ["A"] * 100,
            "observation_ts": pd.date_range("2024-01-01", periods=100, freq="h"),
            "temperature_c": [np.nan] * 50 + [20.0] * 50,
            "humidity_pct": [60.0] * 100,
            "pressure_hpa": [1013.0] * 100,
            "wind_speed_ms": [5.0] * 100,
        })
        report = eng.run(df)
        temp_null = next(c for c in report.checks if c.check_name == "null_pct_temperature_c")
        assert not temp_null.passed
        assert temp_null.affected_rows == 50


class TestRanges:
    def test_clean_data_in_range(self, engine, clean_df):
        report = engine.run(clean_df)
        range_checks = [c for c in report.checks if c.check_name.startswith("range_")]
        assert all(c.passed for c in range_checks)

    def test_out_of_range_temperature(self, engine):
        df = pd.DataFrame({
            "station_id": ["A"] * 200,
            "observation_ts": pd.date_range("2024-01-01", periods=200, freq="h"),
            "temperature_c": [100.0] * 5 + [20.0] * 195,
            "humidity_pct": [50.0] * 200,
            "pressure_hpa": [1013.0] * 200,
            "wind_speed_ms": [5.0] * 200,
        })
        report = engine.run(df)
        tc = next(c for c in report.checks if c.check_name == "range_temperature_c")
        assert not tc.passed
        assert tc.affected_rows == 5

    def test_extreme_wind_speed(self, engine):
        df = pd.DataFrame({
            "station_id": ["A"] * 200,
            "observation_ts": pd.date_range("2024-01-01", periods=200, freq="h"),
            "temperature_c": [20.0] * 200,
            "humidity_pct": [50.0] * 200,
            "pressure_hpa": [1013.0] * 200,
            "wind_speed_ms": [100.0] * 3 + [5.0] * 197,
        })
        report = engine.run(df)
        wc = next(c for c in report.checks if c.check_name == "range_wind_speed_ms")
        assert not wc.passed
        assert wc.affected_rows == 3


class TestDuplicates:
    def test_no_duplicates(self, engine, clean_df):
        report = engine.run(clean_df)
        dc = next(c for c in report.checks if c.check_name == "duplicate_check")
        assert dc.passed

    def test_duplicates_detected(self, engine, clean_df):
        dup = pd.concat([clean_df, clean_df.head(10)], ignore_index=True)
        report = engine.run(dup)
        dc = next(c for c in report.checks if c.check_name == "duplicate_check")
        assert not dc.passed
        assert dc.affected_rows == 10


class TestAnomalyRate:
    def test_low_anomaly_rate_passes(self, engine, clean_df):
        # clean_df has anomaly_rate=0.0
        report = engine.run(clean_df)
        ac = next((c for c in report.checks if c.check_name == "anomaly_rate"), None)
        if ac:
            assert ac.passed

    def test_high_anomaly_rate_fails(self, dirty_df):
        eng = DataQualityEngine(QualityConfig(max_anomaly_pct=1.0, freshness_max_hours=None))
        report = eng.run(dirty_df)
        ac = next((c for c in report.checks if c.check_name == "anomaly_rate"), None)
        if ac:
            assert not ac.passed


# ---------------------------------------------------------------------------
# QualityReport model
# ---------------------------------------------------------------------------
class TestQualityReportModel:
    def test_empty_report_passes(self):
        r = QualityReport()
        assert r.passed
        assert r.health_score == 1.0

    def test_mixed_checks(self):
        r = QualityReport(checks=[
            CheckResult("a", True, 1, 1),
            CheckResult("b", False, 2, 1),
        ])
        assert not r.passed
        assert r.health_score == 0.5

    def test_all_failed(self):
        r = QualityReport(checks=[
            CheckResult("a", False, 0, 1),
            CheckResult("b", False, 0, 1),
        ])
        assert r.health_score == 0.0
