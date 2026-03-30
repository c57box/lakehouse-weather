# Lakehouse Weather Pipeline — Synthetic Data Edition (v2.0)

> **Migration summary:** The Excel-upload ingestion path has been fully replaced by an
> in-process synthetic weather data generator.  No external file dependency remains.

## Architecture

```
┌──────────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Synthetic Generator │────▶│  Data Quality    │────▶│  Ingestion    │
│  (12 stations,       │     │  Engine (10+     │     │  Pipeline     │
│   seasonal/diurnal   │     │   configurable   │     │  Bronze →     │
│   models, anomaly &  │     │   checks)        │     │  Silver →     │
│   null injection)    │     │                  │     │  Gold         │
└──────────────────────┘     └──────────────────┘     └───────────────┘
         │                            │                        │
         └────────────────────────────┼────────────────────────┘
                                      ▼
                          ┌──────────────────────┐
                          │  Monitoring &        │
                          │  Observability       │
                          │  (events, gauges,    │
                          │   health checks)     │
                          └──────────────────────┘
```

## Quick Start

```python
from lakehouse_weather.synthetic_generator import quick_generate

df = quick_generate(n_days=30, n_stations=5, seed=42)
print(df.shape)  # (3600, 19)
```

### Full pipeline run

```python
from lakehouse_weather.ingestion import PipelineConfig, WeatherIngestionPipeline

cfg = PipelineConfig(n_stations=12, n_days=30, seed=42)
pipe = WeatherIngestionPipeline(cfg)
summary = pipe.run()
print(summary)
```

## Project Structure

```
lakehouse_weather/
├── src/lakehouse_weather/
│   ├── __init__.py
│   ├── synthetic_generator.py   # Replaces Excel upload
│   ├── data_quality.py          # Configurable validation engine
│   ├── ingestion.py             # Bronze → Silver → Gold pipeline
│   └── monitoring.py            # Events, metrics, health checks
├── dags/
│   └── weather_pipeline_dag.py  # Airflow DAG (6 tasks, daily)
├── tests/
│   ├── test_synthetic_generator.py
│   ├── test_data_quality.py
│   ├── test_ingestion.py
│   └── test_monitoring.py
├── pyproject.toml
└── README.md
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v --tb=short
```

## Airflow DAG

The DAG `lakehouse_weather_synthetic` runs daily with this task graph:

```
generate_synthetic_data → validate_quality → write_bronze → write_silver → write_gold → publish_metrics
```

- **Deterministic seeds** derived from the logical date enable reproducible backfills.
- **Quality gate** aborts the run if validation fails (configurable).
- **XCom** passes staged file paths between tasks.

## Configuration Reference

| Parameter        | Default | Description                                |
|------------------|---------|--------------------------------------------|
| `n_stations`     | 12      | Number of weather stations to simulate     |
| `n_days`         | 30      | Days of data to generate per run           |
| `seed`           | 42      | RNG seed for reproducibility               |
| `anomaly_rate`   | 0.02    | Fraction of rows with injected anomalies   |
| `null_rate`      | 0.01    | Fraction of randomly nullified cells       |
| `freq_hours`     | 1       | Observation cadence in hours               |
| `fail_on_quality`| True    | Abort pipeline if quality gate fails       |
| `lakehouse_root` | /tmp/…  | Output directory for Parquet layers        |
| `output_format`  | parquet | Output format (parquet or delta)           |
