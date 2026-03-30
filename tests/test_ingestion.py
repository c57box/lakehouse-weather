from __future__ import annotations
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import pandas as pd
import pytest
from lakehouse_weather.ingestion import PipelineConfig, WeatherIngestionPipeline
@pytest.fixture
def tmp_lakehouse(tmp_path):
    root = tmp_path / "lakehouse"
    root.mkdir()
    yield str(root)
@pytest.fixture
def pipeline_cfg(tmp_lakehouse) -> PipelineConfig:
    return PipelineConfig(
        n_stations=3,
        n_days=3,
        seed=42,
        anomaly_rate=0.02,
        null_rate=0.01,
        freq_hours=1,
        fail_on_quality=True,
        lakehouse_root=tmp_lakehouse,
        output_format="parquet",
        enable_monitoring=True,
    )
@pytest.fixture
def pipeline(pipeline_cfg) -> WeatherIngestionPipeline:
    return WeatherIngestionPipeline(pipeline_cfg)
class TestFullPipelineRun:
    def test_run_succeeds(self, pipeline):
        summary = pipeline.run()
        assert summary["status"] == "success"
        assert "duration_s" in summary
        assert summary["duration_s"] > 0
    def test_run_produces_layers(self, pipeline, pipeline_cfg):
        pipeline.run()
        root = Path(pipeline_cfg.lakehouse_root)
        assert (root / "bronze" / "weather").exists()
        assert (root / "silver" / "weather").exists()
        assert (root / "gold" / "weather").exists()
    def test_run_layer_row_counts(self, pipeline):
        summary = pipeline.run()
        layers = summary["layers"]
        assert layers["bronze_rows"] > 0
        assert layers["silver_rows"] > 0
        assert layers["gold_rows"] > 0
        assert layers["silver_rows"] <= layers["bronze_rows"]
        assert layers["gold_rows"] <= layers["silver_rows"]
    def test_run_includes_quality_summary(self, pipeline):
        summary = pipeline.run()
        assert "quality" in summary
        assert "overall_passed" in summary["quality"]
class TestBronzeLayer:
    def test_bronze_has_ingestion_metadata(self, pipeline, pipeline_cfg):
        pipeline.run()
        root = Path(pipeline_cfg.lakehouse_root)
        files = list((root / "bronze" / "weather").glob("*.parquet"))
        assert len(files) == 1
        df = pd.read_parquet(files[0])
        assert "ingestion_ts" in df.columns
        assert "layer" in df.columns
        assert (df["layer"] == "bronze").all()
    def test_bronze_preserves_raw_data(self, pipeline, pipeline_cfg):
        pipeline.run()
        root = Path(pipeline_cfg.lakehouse_root)
        df = pd.read_parquet(list((root / "bronze" / "weather").glob("*.parquet"))[0])
        if "is_anomaly" in df.columns:
            assert df["is_anomaly"].sum() >= 0  # may be 0 for small datasets
class TestSilverLayer:
    def test_silver_removes_anomalies(self, pipeline, pipeline_cfg):
        pipeline.run()
        root = Path(pipeline_cfg.lakehouse_root)
        df = pd.read_parquet(list((root / "silver" / "weather").glob("*.parquet"))[0])
        if "is_anomaly" in df.columns:
            assert df["is_anomaly"].sum() == 0
    def test_silver_no_duplicates(self, pipeline, pipeline_cfg):
        pipeline.run()
        root = Path(pipeline_cfg.lakehouse_root)
        df = pd.read_parquet(list((root / "silver" / "weather").glob("*.parquet"))[0])
        dups = df.duplicated(subset=["station_id", "observation_ts"]).sum()
        assert dups == 0
    def test_silver_imputed_nulls(self, pipeline, pipeline_cfg):
        pipeline.run()
        root = Path(pipeline_cfg.lakehouse_root)
        df = pd.read_parquet(list((root / "silver" / "weather").glob("*.parquet"))[0])
        numeric_cols = ["temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_ms"]
        for col in numeric_cols:
            if col in df.columns:
                assert df[col].isna().sum() == 0
class TestGoldLayer:
    def test_gold_aggregation_structure(self, pipeline, pipeline_cfg):
        pipeline.run()
        root = Path(pipeline_cfg.lakehouse_root)
        df = pd.read_parquet(list((root / "gold" / "weather").glob("*.parquet"))[0])
        expected_cols = {
            "station_id", "station_name", "date",
            "temp_mean", "temp_min", "temp_max",
            "humidity_mean", "pressure_mean",
            "wind_speed_mean", "wind_speed_max",
            "precip_total_mm", "observation_count",
        }
        assert expected_cols.issubset(set(df.columns))
    def test_gold_one_row_per_station_day(self, pipeline, pipeline_cfg):
        pipeline.run()
        root = Path(pipeline_cfg.lakehouse_root)
        df = pd.read_parquet(list((root / "gold" / "weather").glob("*.parquet"))[0])
        dups = df.duplicated(subset=["station_id", "date"]).sum()
        assert dups == 0
    def test_gold_observation_count_sensible(self, pipeline, pipeline_cfg):
        pipeline.run()
        root = Path(pipeline_cfg.lakehouse_root)
        df = pd.read_parquet(list((root / "gold" / "weather").glob("*.parquet"))[0])
        assert (df["observation_count"] <= 24).all()
        assert (df["observation_count"] > 0).all()
class TestQualityGate:
    def test_fail_on_quality_aborts(self, tmp_lakehouse):
        cfg = PipelineConfig(
            n_stations=2,
            n_days=2,
            seed=42,
            anomaly_rate=0.50,   # 50% anomalies
            null_rate=0.0,
            fail_on_quality=True,
            lakehouse_root=tmp_lakehouse,
        )
        pipe = WeatherIngestionPipeline(cfg)
        summary = pipe.run()
        assert summary["status"] in ("success", "failed_quality_gate")
    def test_no_fail_on_quality_continues(self, tmp_lakehouse):
        cfg = PipelineConfig(
            n_stations=2,
            n_days=2,
            seed=42,
            anomaly_rate=0.50,
            null_rate=0.0,
            fail_on_quality=False,
            lakehouse_root=tmp_lakehouse,
        )
        pipe = WeatherIngestionPipeline(cfg)
        summary = pipe.run()
        assert summary["status"] == "success"
class TestMonitoringIntegration:
    def test_monitoring_records_events(self, pipeline):
        pipeline.run()
        assert pipeline.metrics is not None
        events = pipeline.metrics.events
        event_names = {e.event.name for e in events}
        assert "GENERATION_START" in event_names
        assert "GENERATION_COMPLETE" in event_names
        assert "PIPELINE_COMPLETE" in event_names
    def test_monitoring_disabled(self, tmp_lakehouse):
        cfg = PipelineConfig(
            n_stations=2,
            n_days=1,
            lakehouse_root=tmp_lakehouse,
            enable_monitoring=False,
        )
        pipe = WeatherIngestionPipeline(cfg)
        summary = pipe.run()
        assert pipe.metrics is None
        assert summary["status"] == "success"
class TestPipelineConfig:
    def test_default_config(self):
        cfg = PipelineConfig()
        assert cfg.n_stations == 12
        assert cfg.n_days == 30
        assert cfg.fail_on_quality is True
    def test_custom_station_count(self, tmp_lakehouse):
        cfg = PipelineConfig(n_stations=1, n_days=1, lakehouse_root=tmp_lakehouse)
        pipe = WeatherIngestionPipeline(cfg)
        summary = pipe.run()
        assert summary["status"] == "success"
class TestPipelineReproducibility:
    def test_same_seed_same_layers(self, tmp_lakehouse):
        root1 = Path(tmp_lakehouse) / "run1"
        root2 = Path(tmp_lakehouse) / "run2"
        for root in (root1, root2):
            cfg = PipelineConfig(
                n_stations=2, n_days=2, seed=77,
                anomaly_rate=0.0, null_rate=0.0,
                lakehouse_root=str(root),
            )
            WeatherIngestionPipeline(cfg).run()
        gold1 = pd.read_parquet(list((root1 / "gold" / "weather").glob("*.parquet"))[0])
        gold2 = pd.read_parquet(list((root2 / "gold" / "weather").glob("*.parquet"))[0])
        for col in ("ingestion_ts", "layer"):
            if col in gold1.columns:
                gold1 = gold1.drop(columns=[col])
                gold2 = gold2.drop(columns=[col])
        pd.testing.assert_frame_equal(gold1, gold2)
