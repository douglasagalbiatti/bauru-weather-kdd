"""
Módulo de Atualização Incremental Contínua (KDD) – Bauru/SP
Captura automaticamente os novos dados disponíveis na API Open-Meteo via blocos anuais,
valida limites físicos, unifica com o histórico e atualiza as bases processadas.
"""

import sys
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from datetime import datetime, timedelta, timezone as dt_timezone
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.selection import (
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_TIMEZONE,
    DEFAULT_HOURLY_VARS,
    ARCHIVE_API_URL,
)
from src.preprocessing import (
    ensure_temporal_continuity,
    validate_and_clean_physical_bounds,
    handle_missing_values,
)
from src.transformation import build_daily_features, build_monthly_climatology


def get_latest_processed_date(processed_parquet_path: str = "data/processed/bauru_weather_processed.parquet") -> pd.Timestamp:
    """
    Retorna o timestamp mais recente presente na base processada.
    """
    p = Path(processed_parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo processado não encontrado: {processed_parquet_path}")

    df = pd.read_parquet(p)
    return df.index.max()


def fetch_weather_chunk_requests(
    start_date: str,
    end_date: str,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    hourly_vars: Optional[List[str]] = None,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """
    Coleta um bloco de dados meteorológicos via requisição direta HTTP JSON (rápida e resiliente).
    """
    if hourly_vars is None:
        hourly_vars = DEFAULT_HOURLY_VARS

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_vars,
        "timezone": timezone,
    }

    response = requests.get(ARCHIVE_API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    hourly_data = payload.get("hourly", {})
    df_chunk = pd.DataFrame(hourly_data)
    df_chunk["time"] = pd.to_datetime(df_chunk["time"])
    df_chunk = df_chunk.rename(columns={"time": "date"}).set_index("date")
    return df_chunk


def update_weather_pipeline(
    processed_dir: str = "data/processed",
    raw_dir: str = "data/raw",
    reports_dir: str = "reports/tables",
) -> Dict[str, Any]:
    """
    Executa a sincronização incremental fatiada por anos até a data disponível mais recente.
    """
    parquet_path = Path(processed_dir) / "bauru_weather_processed.parquet"
    last_ts = get_latest_processed_date(str(parquet_path))
    print(f"1. Último registro atual na base: {last_ts}")

    # Data inicial do novo lote
    next_start_dt = last_ts + pd.Timedelta(hours=1)
    start_year = next_start_dt.year

    # Data final máxima na API (2 dias atrás)
    max_available_dt = datetime.now() - timedelta(days=2)
    end_year = max_available_dt.year

    if next_start_dt > max_available_dt:
        print("-> A base de dados já está 100% atualizada com a data máxima da API!")
        return {"status": "up_to_date", "last_timestamp": str(last_ts)}

    chunks = []
    print(f"2. Coletando dados incrementais de {next_start_dt.strftime('%Y-%m-%d')} a {max_available_dt.strftime('%Y-%m-%d')}...")

    # Itera por anos para garantir requisições rápidas e sem timeout
    for yr in range(start_year, end_year + 1):
        chunk_start = next_start_dt.strftime("%Y-%m-%d") if yr == start_year else f"{yr}-01-01"
        chunk_end = max_available_dt.strftime("%Y-%m-%d") if yr == end_year else f"{yr}-12-31"

        if pd.to_datetime(chunk_start) > pd.to_datetime(chunk_end):
            continue

        print(f"   Baixando janela: {chunk_start} até {chunk_end}...")
        df_yr = fetch_weather_chunk_requests(chunk_start, chunk_end)
        chunks.append(df_yr)

    if not chunks:
        print("-> Nenhum dado novo a processar.")
        return {"status": "up_to_date"}

    df_new = pd.concat(chunks)
    print(f"   Total de novos registros horários coletados: {len(df_new)}")

    # Salva cópia bruta imutável em data/raw/
    raw_out = Path(raw_dir) / f"bauru_weather_hourly_incremental_{next_start_dt.strftime('%Y%m%d')}_{max_available_dt.strftime('%Y%m%d')}.csv"
    df_new.to_csv(raw_out)
    print(f"   Lote bruto incremental salvo em: {raw_out}")

    print("3. Unificando com a base histórica e aplicando pré-processamento...")
    df_current = pd.read_parquet(parquet_path)
    df_combined = pd.concat([df_current, df_new])

    # Limpeza e validação de limites físicos
    df_cont, temporal_audit = ensure_temporal_continuity(df_combined)
    df_bounded, bounds_audit = validate_and_clean_physical_bounds(df_cont)
    df_clean, imputation_audit = handle_missing_values(df_bounded)

    # Persistência dos dados limpos unificados
    csv_processed = Path(processed_dir) / "bauru_weather_processed.csv"
    df_clean.to_parquet(parquet_path)
    df_clean.to_csv(csv_processed)
    print(f"   Base horária processada atualizada: {df_clean.shape[0]} registros totais.")

    print("4. Atualizando datasets diário e mensal...")
    df_daily = build_daily_features(df_clean)
    df_monthly = build_monthly_climatology(df_daily)

    daily_parquet = Path(processed_dir) / "bauru_weather_daily.parquet"
    daily_csv = Path(processed_dir) / "bauru_weather_daily.csv"
    monthly_parquet = Path(processed_dir) / "bauru_weather_monthly.parquet"
    monthly_csv = Path(processed_dir) / "bauru_weather_monthly.csv"

    df_daily.to_parquet(daily_parquet)
    df_daily.to_csv(daily_csv)
    df_monthly.to_parquet(monthly_parquet)
    df_monthly.to_csv(monthly_csv)

    log_report = {
        "updated_at_utc": datetime.now(dt_timezone.utc).isoformat(),
        "previous_last_timestamp": str(last_ts),
        "new_last_timestamp": str(df_clean.index.max()),
        "new_records_added_hours": len(df_new),
        "total_records_hourly": len(df_clean),
        "total_records_daily": len(df_daily),
        "total_records_monthly": len(df_monthly),
    }

    log_path = Path(reports_dir) / "update_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_report, f, indent=2, ensure_ascii=False)

    print(f"-> Sincronização concluída com sucesso! Novo período coberto: até {df_clean.index.max()}")
    print(f"-> Relatório de atualização salvo em: {log_path}")

    return log_report


if __name__ == "__main__":
    update_weather_pipeline()
