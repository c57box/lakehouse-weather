"""
Airflow DAG — Lakehouse Weather Pipeline (Synthetic Data)
==========================================================
Replaces the former Excel-upload DAG.  There is **no** file-upload task;
instead the ``generate_synthetic_data`` task calls the in-process generator.

DAG structure:
  generate_synthetic_data -> validate_quality -> write_bronze
                                                     |
                                                write_silver
                                                     |
                                                 write_gold
                                                     |
                                              publish_metrics
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# DAG-level settings
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

LAKEHOUSE_ROOT = "/opt/lakehouse"   # swap for ADLS / S3 prefix in prod


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------
def _generate_synthetic_data(**ctx):
    """Generate synthetic weather observations and push to XCom."""
    from lakehouse_weather.synthetic_generator import (
        GeneratorConfig,
        DEFAULT_STATIONS,
        SyntheticWeatherGenerator,
    )

    logical_date: datetime = ctx["logical_date"]
    start = logical_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1) - timedelta(hours=1)

    cfg = GeneratorConfig(
        stations=DEFAULT_STATIONS,
        start_date=start,
        end_date=end,
        freq_hours=1,
        seed=int(start.strftime("%Y%m%d")),
        anomaly_rate=0.02,
        null_rate=0.01,
    )
    gen = SyntheticWeatherGenerator(cfg)
    df = gen.generate()
    path = Path(LAKEHOUSE_ROOT) / "staging" / f"raw_{start:%Y%m%d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logging.info("Staged %d rows -> %s", len(df), path)
    ctx["ti"].xcom_push(key="staged_path", value=str(path))
    ctx["ti"].xcom_push(key="row_count", value=len(df))


def _validate_quality(**ctx):
    """Run data-quality checks on the staged file."""
    import pandas as pd
    from lakehouse_weather.data_quality import DataQualityEngine, QualityConfig

    path = ctx["ti"].xcom_pull(task_ids="generate_synthetic_data", key="staged_path")
    df = pd.read_parquet(path)
    engine = DataQualityEngine(QualityConfig(freshness_max_hours=None))
    report = engine.run(df)
    summary = report.summary_dict()
    ctx["ti"].xcom_push(key="quality_summary", value=json.dumps(summary, default=str))
    if not report.passed:
        raise RuntimeError(f"Quality gate FAILED: {summary['failed_checks']}")
    logging.info("Quality gate PASSED — health score %.2f", report.health_score)


def _write_bronze(**ctx):
    """Persist raw data as the Bronze layer."""
    import pandas as pd

    path = ctx["ti"].xcom_pull(task_ids="generate_synthetic_data", key="staged_path")
    df = pd.read_parquet(path)
    df["layer"] = "bronze"
    df["ingestion_ts"] = datetime.utcnow()
    out = Path(LAKEHOUSE_ROOT) / "bronze" / "weather" / Path(path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    ctx["ti"].xcom_push(key="bronze_path", value=str(out))
    logging.info("Bronze layer -> %s", out)


def _write_silver(**ctx):
    """Clean, deduplicate, and impute — Silver layer."""
    import pandas as pd

    bronze_path = ctx["ti"].xcom_pull(task_ids="write_bronze", key="bronze_path")
    df = pd.read_parquet(bronze_path)
    df = df.drop_duplicates(subset=["station_id", "observation_ts"])
    if "is_anomaly" in df.columns:
        df = df[~df["is_anomaly"]].copy()
    numeric_cols = ["temperature_c", "humidity_pct", "pressure_hpa",
                    "wind_speed_ms", "precipitation_mm", "visibility_km"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df.groupby("station_id")[col].transform(
                lambda s: s.fillna(s.median())
            )
    df["layer"] = "silver"
    out = Path(LAKEHOUSE_ROOT) / "silver" / "weather" / Path(bronze_path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    ctx["ti"].xcom_push(key="silver_path", value=str(out))
    logging.info("Silver layer -> %s  (%d rows)", out, len(df))


def _write_gold(**ctx):
    """Aggregate to daily station-level summaries — Gold layer."""
    import pandas as pd

    silver_path = ctx["ti"].xcom_pull(task_ids="write_silver", key="silver_path")
    df = pd.read_parquet(silver_path)
    df["date"] = pd.to_datetime(df["observation_ts"]).dt.date
    gold = df.groupby(["station_id", "station_name", "date"]).agg(
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
    for c in ["temp_mean", "temp_min", "temp_max", "humidity_mean",
              "pressure_mean", "wind_speed_mean", "wind_speed_max",
              "precip_total_mm"]:
        gold[c] = gold[c].round(2)
    gold["layer"] = "gold"
    out = Path(LAKEHOUSE_ROOT) / "gold" / "weather" / Path(silver_path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(out, index=False)
    logging.info("Gold layer -> %s  (%d rows)", out, len(gold))


def _publish_metrics(**ctx):
    """Log consolidated run metrics."""
    quality = ctx["ti"].xcom_pull(
        task_ids="validate_quality", key="quality_summary"
    )
    row_count = ctx["ti"].xcom_pull(
        task_ids="generate_synthetic_data", key="row_count"
    )
    logging.info(
        "METRICS | rows_generated=%s | quality=%s", row_count, quality
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="lakehouse_weather_synthetic",
    description="Synthetic weather data -> Bronze -> Silver -> Gold pipeline",
    default_args=DEFAULT_ARGS,
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["lakehouse", "weather", "synthetic", "data-engineering"],
    max_active_runs=1,
) as dag:

    generate = PythonOperator(
        task_id="generate_synthetic_data",
        python_callable=_generate_synthetic_data,
    )

    validate = PythonOperator(
        task_id="validate_quality",
        python_callable=_validate_quality,
    )

    bronze = PythonOperator(
        task_id="write_bronze",
        python_callable=_write_bronze,
    )

    silver = PythonOperator(
        task_id="write_silver",
        python_callable=_write_silver,
    )

    gold = PythonOperator(
        task_id="write_gold",
        python_callable=_write_gold,
    )

    metrics = PythonOperator(
        task_id="publish_metrics",
        python_callable=_publish_metrics,
    )

    generate >> validate >> bronze >> silver >> gold >> metrics
