"""
Módulo de Pré-processamento e Limpeza (Etapa 3 - KDD)
Responsável por carregar os dados brutos de data/raw/, validar limites físicos,
tratar consistência temporal, imputar eventuais falhas e persistir em data/processed/.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

PHYSICAL_LIMITS = {
    "temperature_2m": {"min": -5.0, "max": 48.0, "unit": "°C"},
    "relative_humidity_2m": {"min": 0.0, "max": 100.0, "unit": "%"},
    "precipitation": {"min": 0.0, "max": 250.0, "unit": "mm"},
    "wind_speed_10m": {"min": 0.0, "max": 150.0, "unit": "km/h"},
    "surface_pressure": {"min": 850.0, "max": 1050.0, "unit": "hPa"},
    "shortwave_radiation": {"min": 0.0, "max": 1400.0, "unit": "W/m²"},
}


def load_raw_data(raw_csv_path: str) -> pd.DataFrame:
    """
    Carrega o arquivo CSV bruto e converte a coluna temporal para DatetimeIndex.
    Os horários são locais padrão (UTC-03:00), sem informação de fuso no índice.
    """
    df = pd.read_csv(raw_csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def ensure_local_time(df: pd.DataFrame, metadata_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Garante que a carga histórica esteja em horário local.

    Versões antigas de `selection.py` gravavam o índice em UTC (a série começava às 03:00
    do primeiro dia). Se os metadados não indicam horário local e o primeiro registro
    cai exatamente no deslocamento UTC, o índice é corrigido somando o deslocamento
    (-3 h em Bauru). Arquivos já gravados em horário local não são alterados.
    """
    audit: Dict[str, Any] = {"timezone_shift_hours": 0, "legacy_utc_detected": False}
    meta_file = Path(metadata_path)
    if not meta_file.exists() or df.empty:
        return df, audit

    with open(meta_file, encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("time_reference", "").startswith("horario_local"):
        return df, audit

    offset_s = int(metadata.get("timezone_offset_seconds", 0))
    expected_utc_start = pd.Timestamp(metadata["start_date"]) - pd.Timedelta(seconds=offset_s)
    if offset_s != 0 and df.index.min() == expected_utc_start:
        df = df.copy()
        df.index = df.index + pd.Timedelta(seconds=offset_s)
        audit["timezone_shift_hours"] = offset_s / 3600
        audit["legacy_utc_detected"] = True
    return df, audit


def load_incremental_raws(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Carrega todos os lotes incrementais (2025 em diante), já em horário local,
    gravados por `update_weather.py` em data/raw/.
    """
    files = sorted(Path(raw_dir).glob("bauru_weather_hourly_incremental_*.csv"))
    if not files:
        return pd.DataFrame()
    frames = [load_raw_data(str(f)) for f in files]
    df = pd.concat(frames)
    # Se dois lotes cobrirem a mesma hora, vale o mais recente
    return df[~df.index.duplicated(keep="last")].sort_index()


def ensure_temporal_continuity(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remove duplicatas temporais, ordena o índice e garante que não haja lacunas de horas.
    """
    audit = {}
    initial_rows = len(df)
    
    # Remoção de duplicatas de índice
    df = df[~df.index.duplicated(keep="first")].sort_index()
    audit["duplicates_removed"] = initial_rows - len(df)

    # Reindexação para grade temporal estritamente horária
    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="1h",
    )
    missing_timestamps = len(full_idx) - len(df)
    audit["missing_timestamps_inserted"] = missing_timestamps

    df_continuous = df.reindex(full_idx)
    return df_continuous, audit


def validate_and_clean_physical_bounds(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Verifica e trata inconsistências físicas conhecidas.
    """
    df_cleaned = df.copy()
    bounds_audit: Dict[str, Any] = {}

    for col, limits in PHYSICAL_LIMITS.items():
        if col not in df_cleaned.columns:
            continue

        below_min = (df_cleaned[col] < limits["min"]).sum()
        above_max = (df_cleaned[col] > limits["max"]).sum()

        bounds_audit[col] = {
            "out_of_bounds_below": int(below_min),
            "out_of_bounds_above": int(above_max),
        }

        # Trunca limites físicos estritos (UR não pode ser < 0 ou > 100; chuva não pode ser < 0)
        if col == "relative_humidity_2m":
            df_cleaned[col] = df_cleaned[col].clip(lower=0.0, upper=100.0)
        elif col == "precipitation":
            df_cleaned[col] = df_cleaned[col].clip(lower=0.0)
        elif col in ("wind_speed_10m", "shortwave_radiation"):
            df_cleaned[col] = df_cleaned[col].clip(lower=0.0)

    return df_cleaned, bounds_audit


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trata valores faltantes (NaNs):
    - Interpolação linear temporal para variáveis meteorológicas contínuas.
    - Preenchimento com zero para precipitação acumulada pontual faltante.
    """
    missing_before = df.isna().sum().to_dict()
    df_imputed = df.copy()

    # Precipitação: se nula, assume ausência de chuva (0.0 mm)
    if "precipitation" in df_imputed.columns:
        df_imputed["precipitation"] = df_imputed["precipitation"].fillna(0.0)

    # Variáveis contínuas: interpolação temporal com limite de até 3 horas consecutivas
    continuous_cols = [c for c in df_imputed.columns if c != "precipitation"]
    df_imputed[continuous_cols] = df_imputed[continuous_cols].interpolate(
        method="time", limit=3, limit_direction="both"
    )

    # Se ainda restar algum NaN isolado nos extremos, preenche com bfill/ffill
    df_imputed = df_imputed.bfill().ffill()
    missing_after = df_imputed.isna().sum().to_dict()

    imputation_audit = {
        "missing_before": {k: int(v) for k, v in missing_before.items()},
        "missing_after": {k: int(v) for k, v in missing_after.items()},
    }
    return df_imputed, imputation_audit


def run_preprocessing_pipeline(
    raw_path: str = "data/raw/bauru_weather_hourly_2010-01-01_2024-12-31.csv",
    output_dir: str = "data/processed",
    reports_dir: str = "reports/tables",
    include_incremental: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executa o fluxo completo da Etapa 3 e persiste o resultado.
    Une a carga histórica (2010–2024) aos lotes incrementais de data/raw/, para que
    reprocessar a base nunca descarte os dados de 2025 em diante.
    """
    print(f"1. Carregando dados brutos de: {raw_path}")
    df_raw = load_raw_data(raw_path)
    metadata_path = str(raw_path).replace(".csv", "_metadata.json")
    df_raw, tz_audit = ensure_local_time(df_raw, metadata_path)
    if tz_audit["legacy_utc_detected"]:
        print(f"   Carga histórica em UTC detectada: índice deslocado {tz_audit['timezone_shift_hours']:+.0f} h para o horário local.")

    if include_incremental:
        df_inc = load_incremental_raws(str(Path(raw_path).parent))
        if not df_inc.empty:
            print(f"   + {len(df_inc)} registros incrementais ({df_inc.index.min()} a {df_inc.index.max()})")
            df_raw = pd.concat([df_raw, df_inc])

    print("2. Verificando continuidade do eixo temporal...")
    df_cont, temporal_audit = ensure_temporal_continuity(df_raw)

    print("3. Validando limites físicos e consistência...")
    df_bounded, bounds_audit = validate_and_clean_physical_bounds(df_cont)

    print("4. Tratando eventuais dados faltantes...")
    df_clean, imputation_audit = handle_missing_values(df_bounded)

    audit_report = {
        "dataset_shape": {"rows": len(df_clean), "columns": len(df_clean.columns)},
        "date_range": {
            "start": str(df_clean.index.min()),
            "end": str(df_clean.index.max()),
        },
        "timezone_audit": tz_audit,
        "temporal_audit": temporal_audit,
        "physical_bounds_audit": bounds_audit,
        "imputation_audit": imputation_audit,
    }

    # Persistência
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    rep_dir_path = Path(reports_dir)
    rep_dir_path.mkdir(parents=True, exist_ok=True)

    parquet_file = out_dir_path / "bauru_weather_processed.parquet"
    csv_file = out_dir_path / "bauru_weather_processed.csv"
    audit_file = rep_dir_path / "preprocessing_summary.json"

    df_clean.to_parquet(parquet_file)
    df_clean.to_csv(csv_file)

    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    print(f"-> Dados processados salvos em: {parquet_file} e {csv_file}")
    print(f"-> Relatório de auditoria salvo em: {audit_file}")
    return df_clean, audit_report


if __name__ == "__main__":
    run_preprocessing_pipeline()
