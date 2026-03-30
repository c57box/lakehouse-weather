"""
Data Quality Engine
====================
Runs configurable validation rules against weather DataFrames.
Returns a structured report with pass/fail per check, affected row counts,
and an overall health score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    check_name: str
    passed: bool
    metric_value: float
    threshold: float
    affected_rows: int = 0
    details: str = ""


@dataclass
class QualityReport:
    run_ts: datetime = field(default_factory=datetime.utcnow)
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def health_score(self) -> float:
        if not self.checks:
            return 1.0
        return sum(c.passed for c in self.checks) / len(self.checks)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "run_ts": self.run_ts.isoformat(),
            "overall_passed": self.passed,
            "health_score": round(self.health_score, 4),
            "total_checks": len(self.checks),
            "failed_checks": [c.check_name for c in self.checks if not c.passed],
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for c in self.checks:
            rows.append({
                "check_name": c.check_name,
                "passed": c.passed,
                "metric_value": c.metric_value,
                "threshold": c.threshold,
                "affected_rows": c.affected_rows,
                "details": c.details,
            })
        return pd.DataFrame(rows)


@dataclass
class QualityConfig:
    """Thresholds and toggles for every built-in check."""
    max_null_pct: float = 5.0
    temp_range: tuple = (-60.0, 55.0)
    humidity_range: tuple = (0.0, 100.0)
    pressure_range: tuple = (870.0, 1084.0)
    wind_speed_max: float = 75.0
    max_duplicate_pct: float = 1.0
    min_row_count: int = 100
    max_anomaly_pct: float = 5.0
    required_columns: List[str] = field(default_factory=lambda: [
        "station_id", "observation_ts", "temperature_c",
        "humidity_pct", "pressure_hpa", "wind_speed_ms",
    ])
    freshness_max_hours: Optional[float] = 48.0


class DataQualityEngine:
    """Execute all configured checks against a weather DataFrame."""

    def __init__(self, config: Optional[QualityConfig] = None):
        self.cfg = config or QualityConfig()

    def run(self, df: pd.DataFrame) -> QualityReport:
        report = QualityReport()
        report.checks.append(self._check_row_count(df))
        report.checks.append(self._check_schema(df))
        report.checks.extend(self._check_nulls(df))
        report.checks.extend(self._check_ranges(df))
        report.checks.append(self._check_duplicates(df))
        if "is_anomaly" in df.columns:
            report.checks.append(self._check_anomaly_rate(df))
        if "observation_ts" in df.columns and self.cfg.freshness_max_hours:
            report.checks.append(self._check_freshness(df))
        logger.info("Quality report: %s", report.summary_dict())
        return report

    def _check_row_count(self, df: pd.DataFrame) -> CheckResult:
        n = len(df)
        passed = n >= self.cfg.min_row_count
        return CheckResult(
            check_name="row_count_minimum",
            passed=passed,
            metric_value=n,
            threshold=self.cfg.min_row_count,
            details=f"{n} rows present (min {self.cfg.min_row_count})",
        )

    def _check_schema(self, df: pd.DataFrame) -> CheckResult:
        missing = [c for c in self.cfg.required_columns if c not in df.columns]
        return CheckResult(
            check_name="schema_completeness",
            passed=len(missing) == 0,
            metric_value=len(df.columns),
            threshold=len(self.cfg.required_columns),
            details=f"missing columns: {missing}" if missing else "all required columns present",
        )

    def _check_nulls(self, df: pd.DataFrame) -> List[CheckResult]:
        results = []
        for col in self.cfg.required_columns:
            if col not in df.columns:
                continue
            null_pct = df[col].isna().mean() * 100
            results.append(CheckResult(
                check_name=f"null_pct_{col}",
                passed=null_pct <= self.cfg.max_null_pct,
                metric_value=round(null_pct, 4),
                threshold=self.cfg.max_null_pct,
                affected_rows=int(df[col].isna().sum()),
                details=f"{col}: {null_pct:.2f}% null",
            ))
        return results

    def _check_ranges(self, df: pd.DataFrame) -> List[CheckResult]:
        checks_map = {
            "temperature_c": self.cfg.temp_range,
            "humidity_pct": self.cfg.humidity_range,
            "pressure_hpa": self.cfg.pressure_range,
        }
        results = []
        for col, (lo, hi) in checks_map.items():
            if col not in df.columns:
                continue
            oob = ((df[col] < lo) | (df[col] > hi)).sum()
            oob_pct = oob / len(df) * 100
            results.append(CheckResult(
                check_name=f"range_{col}",
                passed=oob == 0,
                metric_value=round(oob_pct, 4),
                threshold=0.0,
                affected_rows=int(oob),
                details=f"{col}: {oob} rows outside [{lo}, {hi}]",
            ))
        if "wind_speed_ms" in df.columns:
            oob = (df["wind_speed_ms"] > self.cfg.wind_speed_max).sum()
            results.append(CheckResult(
                check_name="range_wind_speed_ms",
                passed=oob == 0,
                metric_value=int(oob),
                threshold=0,
                affected_rows=int(oob),
                details=f"wind_speed_ms: {oob} rows > {self.cfg.wind_speed_max}",
            ))
        return results

    def _check_duplicates(self, df: pd.DataFrame) -> CheckResult:
        key_cols = ["station_id", "observation_ts"]
        if not all(c in df.columns for c in key_cols):
            return CheckResult("duplicate_check", True, 0, 0, details="key columns missing — skipped")
        dup_count = df.duplicated(subset=key_cols).sum()
        dup_pct = dup_count / len(df) * 100
        return CheckResult(
            check_name="duplicate_check",
            passed=dup_pct <= self.cfg.max_duplicate_pct,
            metric_value=round(dup_pct, 4),
            threshold=self.cfg.max_duplicate_pct,
            affected_rows=int(dup_count),
            details=f"{dup_count} duplicate (station, ts) pairs ({dup_pct:.2f}%)",
        )

    def _check_anomaly_rate(self, df: pd.DataFrame) -> CheckResult:
        anomaly_pct = df["is_anomaly"].mean() * 100
        return CheckResult(
            check_name="anomaly_rate",
            passed=anomaly_pct <= self.cfg.max_anomaly_pct,
            metric_value=round(anomaly_pct, 4),
            threshold=self.cfg.max_anomaly_pct,
            affected_rows=int(df["is_anomaly"].sum()),
            details=f"{anomaly_pct:.2f}% anomalies (max {self.cfg.max_anomaly_pct}%)",
        )

    def _check_freshness(self, df: pd.DataFrame) -> CheckResult:
        newest = pd.to_datetime(df["observation_ts"]).max()
        age_hours = (datetime.utcnow() - newest).total_seconds() / 3600
        return CheckResult(
            check_name="data_freshness",
            passed=age_hours <= self.cfg.freshness_max_hours,
            metric_value=round(age_hours, 2),
            threshold=self.cfg.freshness_max_hours,
            details=f"newest record is {age_hours:.1f}h old (max {self.cfg.freshness_max_hours}h)",
        )
