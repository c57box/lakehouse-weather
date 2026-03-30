"""Synthetic Weather Data Generator - generates realistic weather observations."""
from __future__ import annotations
import hashlib, math, random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

@dataclass
class StationProfile:
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float
    temp_offset_c: float = 0.0
    humidity_offset_pct: float = 0.0

DEFAULT_STATIONS: List[StationProfile] = [
    StationProfile("WS-001", "Seattle-Tacoma",   47.45, -122.31, 130, -1.5, 8),
    StationProfile("WS-002", "Phoenix-Mesa",     33.43, -112.02, 331, 8.0, -20),
    StationProfile("WS-003", "Miami-Dade",       25.79, -80.29,    3, 6.0, 12),
    StationProfile("WS-004", "Denver-Intl",      39.85, -104.67, 1655, -3.0, -10),
    StationProfile("WS-005", "Chicago-OHare",    41.97, -87.90,  201, -0.5, 2),
    StationProfile("WS-006", "Anchorage-Intl",   61.17, -150.00,  46, -12.0, 5),
    StationProfile("WS-007", "Honolulu-HNL",     21.32, -157.92,   4, 10.0, 10),
    StationProfile("WS-008", "Dallas-FtWorth",   32.90, -97.04,  171, 4.0, -5),
    StationProfile("WS-009", "NewYork-JFK",      40.64, -73.78,    4, 1.0, 3),
    StationProfile("WS-010", "Minneapolis-StPl", 44.88, -93.22,  256, -4.0, 0),
    StationProfile("WS-011", "SanFran-SFO",      37.62, -122.38,   4, 2.5, 6),
    StationProfile("WS-012", "Atlanta-Harts",    33.64, -84.43,  313, 3.0, 4),
]

@dataclass
class GeneratorConfig:
    stations: List[StationProfile] = field(default_factory=lambda: list(DEFAULT_STATIONS))
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2024, 12, 31, 23, 0))
    freq_hours: int = 1
    seed: int = 42
    anomaly_rate: float = 0.02
    null_rate: float = 0.01
    include_metadata_cols: bool = True

class SyntheticWeatherGenerator:
    _BASE_TEMP_MEAN = 15.0
    _BASE_TEMP_AMP = 15.0
    _DIURNAL_AMP = 5.0
    _PRESSURE_MEAN = 1013.25
    _PRESSURE_STD = 8.0
    _WIND_SHAPE = 2.0
    _WIND_SCALE = 5.0

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.cfg = config or GeneratorConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        random.seed(self.cfg.seed)
        self._batch_id = self._make_batch_id()

    def generate(self) -> pd.DataFrame:
        timestamps = self._make_timestamps()
        frames: list[pd.DataFrame] = []
        for station in self.cfg.stations:
            frames.append(self._generate_station(station, timestamps))
        combined = pd.concat(frames, ignore_index=True)
        combined = self._inject_anomalies(combined)
        combined = self._inject_nulls(combined)
        if self.cfg.include_metadata_cols:
            combined = self._add_metadata(combined)
        return combined.sort_values(["observation_ts", "station_id"]).reset_index(drop=True)

    def generate_spark(self, spark):
        return spark.createDataFrame(self.generate())

    def row_count_estimate(self) -> int:
        return len(self._make_timestamps()) * len(self.cfg.stations)

    def _make_batch_id(self) -> str:
        seed_bytes = f"{self.cfg.seed}-{self.cfg.start_date}-{self.cfg.end_date}".encode()
        return hashlib.sha256(seed_bytes).hexdigest()[:12]

    def _make_timestamps(self) -> List[datetime]:
        ts: list[datetime] = []
        cur = self.cfg.start_date
        while cur <= self.cfg.end_date:
            ts.append(cur)
            cur += timedelta(hours=self.cfg.freq_hours)
        return ts

    def _generate_station(self, station: StationProfile, timestamps: List[datetime]) -> pd.DataFrame:
        n = len(timestamps)
        day_of_year = np.array([t.timetuple().tm_yday for t in timestamps], dtype=float)
        hour_of_day = np.array([t.hour + t.minute / 60 for t in timestamps], dtype=float)
        seasonal = self._BASE_TEMP_MEAN + self._BASE_TEMP_AMP * np.sin(2 * math.pi * (day_of_year - 80) / 365)
        diurnal = self._DIURNAL_AMP * np.sin(2 * math.pi * (hour_of_day - 6) / 24)
        temperature_c = seasonal + diurnal + station.temp_offset_c + self._rng.normal(0, 1.5, n)
        base_humidity = 60 + station.humidity_offset_pct
        humidity_seasonal = -10 * np.sin(2 * math.pi * (day_of_year - 80) / 365)
        humidity = np.clip(base_humidity + humidity_seasonal + self._rng.normal(0, 5, n), 5, 100)
        elev_adj = -station.elevation_m * 0.12
        pressure = self._PRESSURE_MEAN + elev_adj + self._rng.normal(0, self._PRESSURE_STD, n)
        wind_speed = self._rng.weibull(self._WIND_SHAPE, n) * self._WIND_SCALE
        wind_dir = self._rng.uniform(0, 360, n)
        precip_flag = self._rng.random(n) < (0.15 + 0.10 * np.sin(2 * math.pi * (day_of_year - 100) / 365))
        precipitation_mm = np.where(precip_flag, self._rng.exponential(2.5, n), 0.0)
        visibility = np.clip(self._rng.normal(15, 4, n), 0.1, 50)
        cloud_cover = np.clip(self._rng.normal(4, 2, n).astype(int), 0, 8)
        solar_elev = np.clip(np.sin(2 * math.pi * (hour_of_day - 6) / 24), 0, 1)
        uv_index = np.clip((solar_elev * 11 * (1 - cloud_cover / 16) + self._rng.normal(0, 0.5, n)).round(1), 0, 15)
        return pd.DataFrame({
            "station_id": station.station_id, "station_name": station.name,
            "latitude": station.latitude, "longitude": station.longitude,
            "elevation_m": station.elevation_m, "observation_ts": timestamps,
            "temperature_c": np.round(temperature_c, 2), "humidity_pct": np.round(humidity, 2),
            "pressure_hpa": np.round(pressure, 2), "wind_speed_ms": np.round(wind_speed, 2),
            "wind_direction_deg": np.round(wind_dir, 1), "precipitation_mm": np.round(precipitation_mm, 2),
            "visibility_km": np.round(visibility, 2), "cloud_cover_oktas": cloud_cover, "uv_index": uv_index,
        })

    def _inject_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        n_anomalies = int(n * self.cfg.anomaly_rate)
        df = df.copy()
        df["is_anomaly"] = False
        if n_anomalies == 0:
            return df
        idx = self._rng.choice(n, size=n_anomalies, replace=False)
        anomaly_type = self._rng.choice(
            ["spike_temp", "neg_pressure", "extreme_wind", "impossible_humidity"], size=n_anomalies)
        for i, atype in zip(idx, anomaly_type):
            if atype == "spike_temp":
                df.loc[i, "temperature_c"] = self._rng.uniform(55, 70)
            elif atype == "neg_pressure":
                df.loc[i, "pressure_hpa"] = self._rng.uniform(-50, 0)
            elif atype == "extreme_wind":
                df.loc[i, "wind_speed_ms"] = self._rng.uniform(80, 120)
            elif atype == "impossible_humidity":
                df.loc[i, "humidity_pct"] = self._rng.uniform(110, 200)
        df.loc[idx, "is_anomaly"] = True
        return df

    def _inject_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        nullable_cols = ["temperature_c", "humidity_pct", "pressure_hpa",
                         "wind_speed_ms", "precipitation_mm", "visibility_km"]
        n_nulls_per_col = int(n * self.cfg.null_rate)
        if n_nulls_per_col == 0:
            return df
        df = df.copy()
        for col in nullable_cols:
            df.loc[self._rng.choice(n, size=n_nulls_per_col, replace=False), col] = np.nan
        return df

    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ingestion_ts"] = datetime.utcnow()
        df["data_source"] = "synthetic_generator_v2"
        df["batch_id"] = self._batch_id
        return df

def quick_generate(n_days: int = 30, n_stations: int = 5, seed: int = 42, anomaly_rate: float = 0.02) -> pd.DataFrame:
    cfg = GeneratorConfig(
        stations=DEFAULT_STATIONS[:n_stations],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 1) + timedelta(days=n_days) - timedelta(hours=1),
        seed=seed, anomaly_rate=anomaly_rate)
    return SyntheticWeatherGenerator(cfg).generate()

if __name__ == "__main__":
    df = quick_generate()
    print(f"Generated {len(df):,} rows  |  columns: {list(df.columns)}")
    print(df.head(10).to_string(index=False))
