"""
Ingestion Pipeline
===================
Orchestrates:
  1. Synthetic data generation
  2. Data quality validation
  3. Bronze > Silver > Gold Lakehouse layer writes
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from lakehouse_weather.data_quality import DataQualityEngine, QualityConfig, QualityReport
from lakehouse_weather.monitoring import MetricsCollector, PipelineEvent
from lakehouse_weather.synthetic_generator import GeneratorConfig, SyntheticWeatherGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Runtime configuration for a single pipeline run."""
    n_stations: int = 12
    n_days: int = 30
    seed: int = 42
    anomaly_rate: float = 0.02
    null_rate: float = 0.01
    freq_hours: int = 1
    fail_on_quality: bool = True
    lakehouse_root: str = "/tmp/lakehouse"
    output_format: str = "parquet"
    enable_monitoring: bool = True


class WeatherIngestionPipeline:
    """End-to-end pipeline: generate > validate > write layers."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self.metrics = MetricsCollector() if self.cfg.enable_monitoring else None
        self._quality_engine = DataQualityEngine(
            QualityConfig(freshness_max_hours=None)
        )

    def run(self) -> Dict[str, Any]:
        """Execute a full pipeline run. Returns a summary dict."""
        run_start = datetime.utcnow()
        summary: Dict[str, Any] = {"run_start": run_start.isoformat(), "status": "running"}
        try:
            self._emit(PipelineEvent.GENERATION_START)
            gen_cfg = self._build_generator_config()
            generator = SyntheticWeatherGenerator(gen_cfg)
            raw_df = generator.generate()
            self._emit(PipelineEvent.GENERATION_COMPLETE, {"rows": len(raw_df)})
            logger.info("Generated %d raw rows", len(raw_df))

            self._emit(PipelineEvent.QUALITY_CHECK_START)
            quality_report = self._quality_engine.run(raw_df)
            summary["quality"] = quality_report.summary_dict()
            if not quality_report.passed and self.cfg.fail_on_quality:
                self._emit(PipelineEvent.QUALITY_CHECK_FAILED, summary["quality"])
                summary["status"] = "failed_quality_gate"
                logger.error("Quality gate FAILED -- aborting pipeline.")
                return summary
            self._emit(PipelineEvent.QUALITY_CHECK_PASSED, summary["quality"])

            bronze_df = self._write_bronze(raw_df)
            silver_df = self._write_silver(bronze_df)
            gold_df = self._write_gold(silver_df)
            summary["layers"] = {
                "bronze_rows": len(bronze_df),
                "silver_rows": len(silver_df),
                "gold_rows": len(gold_df),
            }
            summary["status"] = "success"
            run_end = datetime.utcnow()
            summary["duration_s"] = (run_end - run_start).total_seconds()
            self._emit(PipelineEvent.PIPELINE_COMPLETE, summary)
        except Exception as exc:
            summary["status"] = "error"
            summary["error"] = str(exc)
            self._emit(PipelineEvent.PIPELINE_ERROR, {"error": str(exc)})
            logger.exception("Pipeline run failed")
            raise
        return summary

    def _write_bronze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bronze = raw data with ingestion metadata."""
        df = df.copy()
        df["ingestion_ts"] = datetime.utcnow()
        df["layer"] = "bronze"
        path = Path(self.cfg.lakehouse_root) / "bronze" / "weather"
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"weather_bronze.{self.cfg.output_format}"
        df.to_parquet(out, index=False)
        logger.info("Bronze written: %s  (%d rows)", out, len(df))
        return df

    def _write_silver(self, bronze_df: pd.DataFrame) -> pd.DataFrame:
        """Silver = cleaned, deduplicated, anomaly-flagged data."""
        df = bronze_df.copy()
        df = df.drop_duplicates(subset=["station_id", "observation_ts"])
        if "is_anomaly" in df.columns:
            df = df[~df["is_anomaly"]].copy()
        numeric_cols = [
            "temperature_c", "humidity_pct", "pressure_hpa",
            "wind_speed_ms", "precipitation_mm", "visibility_km",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df.groupby("station_id")[col].transform(
                    lambda s: s.fillna(s.median())
                )
        df["layer"] = "silver"
        path = Path(self.cfg.lakehouse_root) / "silver" / "weather"
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"weather_silver.{self.cfg.output_format}"
        df.to_parquet(out, index=False)
        logger.info("Silver written: %s  (%d rows)", out, len(df))
        return df

    def _write_gold(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """Gold = daily station-level aggregates for analytics."""
        df = silver_df.copy()
        df["date"] = pd.to_datetime(df["observation_ts"]).dt.date
        agg = df.groupby(["station_id", "station_name", "date"]).agg(
            temp_mean=("temperature_c", "mean"),
            temp_min=("temperature_c", "min"),
            temp_max=("temperature_c", "max"),
            humidity_mean=("humidity_pct", "mean"),
            pressure_mean=("pressure_hpa", "mean"),
            wind_speed_mean=("wind_speed_ms", "mean"),
            wind_speed_max=("wind_speed_ms", "max"),
            precip_total_mm=("precipitation_mm", "sum"),
            observation_count=("temperature_c", "count"),
        ).reset_index()
        for col in ["temp_mean", "temp_min", "temp_max", "humidity_mean",
                     "pressure_mean", "wind_speed_mean", "wind_speed_max",
                     "precip_total_mm"]:
            agg[col] = agg[col].round(2)
        agg["layer"] = "gold"
        path = Path(self.cfg.lakehouse_root) / "gold" / "weather"
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"weather_gold.{self.cfg.output_format}"
        agg.to_parquet(out, index=False)
        logger.info("Gold written: %s  (%d rows)", out, len(agg))
        return agg

    def _build_generator_config(self) -> GeneratorConfig:
        from lakehouse_weather.synthetic_generator import DEFAULT_STATIONS
        return GeneratorConfig(
            stations=DEFAULT_STATIONS[: self.cfg.n_stations],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1) + timedelta(days=self.cfg.n_days) - timedelta(hours=1),
            freq_hours=self.cfg.freq_hours,
            seed=self.cfg.seed,
            anomaly_rate=self.cfg.anomaly_rate,
            null_rate=self.cfg.null_rate,
        )

    def _emit(self, event: PipelineEvent, payload: Optional[Dict] = None):
        if self.metrics:
            self.metrics.record(event, payload or {})
