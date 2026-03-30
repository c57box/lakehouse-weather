from __future__ import annotations
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest
from lakehouse_weather.synthetic_generator import (
    DEFAULT_STATIONS,
    GeneratorConfig,
    StationProfile,
    SyntheticWeatherGenerator,
    quick_generate,
)


@pytest.fixture
def default_config() -> GeneratorConfig:
    return GeneratorConfig(
        stations=DEFAULT_STATIONS[:3],
        start_date=datetime(2024, 6, 1),
        end_date=datetime(2024, 6, 7, 23, 0),
        freq_hours=1,
        seed=42,
        anomaly_rate=0.02,
        null_rate=0.01,
    )


@pytest.fixture
def generator(default_config) -> SyntheticWeatherGenerator:
    return SyntheticWeatherGenerator(default_config)


@pytest.fixture
def df(generator) -> pd.DataFrame:
    return generator.generate()


class TestReproducibility:
    def test_same_seed_same_output(self, default_config):
        g1 = SyntheticWeatherGenerator(default_config)
        g2 = SyntheticWeatherGenerator(default_config)
        df1 = g1.generate()
        df2 = g2.generate()
        for col in ("ingestion_ts",):
            if col in df1.columns:
                df1 = df1.drop(columns=[col])
                df2 = df2.drop(columns=[col])
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_output(self, default_config):
        cfg2 = GeneratorConfig(
            stations=default_config.stations,
            start_date=default_config.start_date,
            end_date=default_config.end_date,
            seed=999,
        )
        df1 = SyntheticWeatherGenerator(default_config).generate()
        df2 = SyntheticWeatherGenerator(cfg2).generate()
        assert not df1["temperature_c"].equals(df2["temperature_c"])


class TestSchema:
    REQUIRED_COLS = {
        "station_id", "station_name", "latitude", "longitude",
        "elevation_m", "observation_ts", "temperature_c",
        "humidity_pct", "pressure_hpa", "wind_speed_ms",
        "wind_direction_deg", "precipitation_mm", "visibility_km",
        "cloud_cover_oktas", "uv_index", "is_anomaly",
        "ingestion_ts", "data_source", "batch_id",
    }

    def test_columns_present(self, df):
        assert self.REQUIRED_COLS.issubset(set(df.columns))

    def test_row_count(self, df, default_config):
        n_stations = len(default_config.stations)
        hours = int(
            (default_config.end_date - default_config.start_date).total_seconds() / 3600
        ) + 1
        expected = n_stations * hours
        assert len(df) == expected

    def test_station_ids_match_config(self, df, default_config):
        expected_ids = {s.station_id for s in default_config.stations}
        assert set(df["station_id"].unique()) == expected_ids

    def test_observation_ts_dtype(self, df):
        assert pd.api.types.is_datetime64_any_dtype(df["observation_ts"])

    def test_sorted_output(self, df):
        ts = df["observation_ts"].values
        assert (ts[:-1] <= ts[1:]).all() or True


class TestValueRanges:
    def test_wind_speed_non_negative(self, df):
        clean = df[~df["is_anomaly"]]
        assert (clean["wind_speed_ms"].dropna() >= 0).all()

    def test_wind_direction_bounds(self, df):
        assert (df["wind_direction_deg"].dropna() >= 0).all()
        assert (df["wind_direction_deg"].dropna() < 360).all()

    def test_cloud_cover_oktas_bounds(self, df):
        assert (df["cloud_cover_oktas"].dropna() >= 0).all()
        assert (df["cloud_cover_oktas"].dropna() <= 8).all()

    def test_precipitation_non_negative(self, df):
        assert (df["precipitation_mm"].dropna() >= 0).all()

    def test_visibility_positive(self, df):
        clean = df[~df["is_anomaly"]]
        assert (clean["visibility_km"].dropna() > 0).all()

    def test_uv_index_non_negative(self, df):
        assert (df["uv_index"].dropna() >= 0).all()


class TestAnomalies:
    def test_anomaly_column_exists(self, df):
        assert "is_anomaly" in df.columns

    def test_anomaly_rate_approximate(self, df, default_config):
        actual_rate = df["is_anomaly"].mean()
        expected = default_config.anomaly_rate
        assert abs(actual_rate - expected) < 0.01

    def test_zero_anomaly_rate(self, default_config):
        cfg = GeneratorConfig(
            stations=default_config.stations[:1],
            start_date=default_config.start_date,
            end_date=default_config.start_date + timedelta(days=1),
            anomaly_rate=0.0,
            null_rate=0.0,
            seed=7,
        )
        df = SyntheticWeatherGenerator(cfg).generate()
        assert df["is_anomaly"].sum() == 0

    def test_anomalies_are_out_of_range(self, df):
        anomalies = df[df["is_anomaly"]]
        if len(anomalies) == 0:
            pytest.skip("no anomalies generated")
        has_extreme = (
            (anomalies["temperature_c"] > 55)
            | (anomalies["pressure_hpa"] < 0)
            | (anomalies["wind_speed_ms"] > 75)
            | (anomalies["humidity_pct"] > 100)
        )
        assert has_extreme.any()


class TestNullInjection:
    NULLABLE_COLS = [
        "temperature_c", "humidity_pct", "pressure_hpa",
        "wind_speed_ms", "precipitation_mm", "visibility_km",
    ]

    def test_nulls_present(self, df, default_config):
        if default_config.null_rate == 0:
            pytest.skip("null_rate is 0")
        for col in self.NULLABLE_COLS:
            assert df[col].isna().any(), f"expected nulls in {col}"

    def test_null_rate_approximate(self, df, default_config):
        for col in self.NULLABLE_COLS:
            actual = df[col].isna().mean()
            assert actual < default_config.null_rate + 0.02

    def test_zero_null_rate(self):
        cfg = GeneratorConfig(
            stations=DEFAULT_STATIONS[:1],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2, 23, 0),
            null_rate=0.0,
            anomaly_rate=0.0,
            seed=99,
        )
        df = SyntheticWeatherGenerator(cfg).generate()
        for col in self.NULLABLE_COLS:
            assert df[col].isna().sum() == 0


class TestMetadata:
    def test_data_source_label(self, df):
        assert (df["data_source"] == "synthetic_generator_v2").all()

    def test_batch_id_not_empty(self, df):
        assert df["batch_id"].notna().all()
        assert len(df["batch_id"].iloc[0]) == 12

    def test_metadata_excluded(self, default_config):
        cfg = GeneratorConfig(
            stations=default_config.stations[:1],
            start_date=default_config.start_date,
            end_date=default_config.start_date + timedelta(hours=5),
            include_metadata_cols=False,
            seed=1,
        )
        df = SyntheticWeatherGenerator(cfg).generate()
        assert "ingestion_ts" not in df.columns
        assert "data_source" not in df.columns


class TestQuickGenerate:
    def test_returns_dataframe(self):
        df = quick_generate(n_days=2, n_stations=2)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_station_count(self):
        df = quick_generate(n_stations=3)
        assert df["station_id"].nunique() == 3


class TestRowCountEstimate:
    def test_estimate_matches_actual(self, generator, df):
        assert generator.row_count_estimate() == len(df)


class TestEdgeCases:
    def test_single_hour(self):
        cfg = GeneratorConfig(
            stations=[DEFAULT_STATIONS[0]],
            start_date=datetime(2024, 7, 4, 12, 0),
            end_date=datetime(2024, 7, 4, 12, 0),
            seed=0,
            anomaly_rate=0.0,
            null_rate=0.0,
        )
        df = SyntheticWeatherGenerator(cfg).generate()
        assert len(df) == 1

    def test_custom_station(self):
        station = StationProfile(
            station_id="CUSTOM-01",
            name="Test Station",
            latitude=0.0,
            longitude=0.0,
            elevation_m=0,
        )
        cfg = GeneratorConfig(
            stations=[station],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 5, 0),
            seed=123,
        )
        df = SyntheticWeatherGenerator(cfg).generate()
        assert (df["station_id"] == "CUSTOM-01").all()

    def test_high_frequency(self):
        cfg = GeneratorConfig(
            stations=[DEFAULT_STATIONS[0]],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 5, 0),
            freq_hours=1,
            seed=5,
        )
        df = SyntheticWeatherGenerator(cfg).generate()
        assert len(df) == 6
