"""
Módulo de Seleção (Etapa 2 - KDD)
Responsável pela extração dos dados brutos históricos da API Open-Meteo
para o município de Bauru/SP e persistência imutável em data/raw/.
"""

import json
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Configurações padrão do estudo
DEFAULT_LATITUDE = -22.3147
DEFAULT_LONGITUDE = -49.0608
DEFAULT_TIMEZONE = "America/Sao_Paulo"
DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2024-12-31"

DEFAULT_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
    "shortwave_radiation",
]

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"


def create_api_client(
    cache_path: str = ".cache",
    retries: int = 5,
    backoff_factor: float = 0.5,
) -> openmeteo_requests.Client:
    """
    Inicializa o cliente Open-Meteo configurado com sessão de cache e retry.
    """
    cache_session = requests_cache.CachedSession(cache_path, expire_after=-1)
    retry_session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)
    return openmeteo_requests.Client(session=retry_session)


def fetch_raw_weather_data(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    hourly_vars: Optional[List[str]] = None,
    timezone: str = DEFAULT_TIMEZONE,
    client: Optional[openmeteo_requests.Client] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Realiza a requisição à API Histórica Open-Meteo e retorna o DataFrame bruto
    juntamente com o dicionário de metadados retornados pelo modelo de reanálise.
    """
    if hourly_vars is None:
        hourly_vars = DEFAULT_HOURLY_VARS

    if client is None:
        client = create_api_client()

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_vars,
        "timezone": timezone,
    }

    responses = client.weather_api(ARCHIVE_API_URL, params=params)
    response = responses[0]

    # Metadados de rastreabilidade
    metadata = {
        "requested_latitude": latitude,
        "requested_longitude": longitude,
        "grid_latitude": response.Latitude(),
        "grid_longitude": response.Longitude(),
        "grid_elevation_m": response.Elevation(),
        "timezone": timezone,
        "timezone_offset_seconds": response.UtcOffsetSeconds(),
        "start_date": start_date,
        "end_date": end_date,
        "hourly_variables": hourly_vars,
        "fetched_at_utc": datetime.now(dt_timezone.utc).isoformat(),
        "data_source": "Open-Meteo Historical Weather API (ERA5/ERA5-Land)",
    }

    hourly = response.Hourly()

    # Reconstrução do eixo temporal
    time_series = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=False),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=False),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    data: Dict[str, Any] = {"date": time_series}

    for idx, var_name in enumerate(hourly_vars):
        data[var_name] = hourly.Variables(idx).ValuesAsNumpy()

    df_raw = pd.DataFrame(data=data)
    return df_raw, metadata


def save_raw_dataset(
    df_raw: pd.DataFrame,
    metadata: Dict[str, Any],
    output_dir: str = "data/raw",
    file_prefix: str = "bauru_weather_hourly",
) -> tuple[Path, Path]:
    """
    Persiste os dados brutos em formato Parquet/CSV e os metadados em JSON.
    Garante integridade e imutabilidade do dado bruto.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / f"{file_prefix}_{metadata['start_date']}_{metadata['end_date']}.csv"
    json_path = out_path / f"{file_prefix}_{metadata['start_date']}_{metadata['end_date']}_metadata.json"

    df_raw.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


if __name__ == "__main__":
    print("Iniciando coleta da Etapa 2 (Seleção)...")
    df, meta = fetch_raw_weather_data()
    csv_file, json_file = save_raw_dataset(df, meta)
    print(f"Dados brutos salvos com sucesso em: {csv_file}")
    print(f"Metadados registrados em: {json_file}")
    print(f"Dimensão coletada: {df.shape[0]} registros x {df.shape[1]} colunas")
