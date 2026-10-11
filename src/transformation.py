"""
Módulo de Transformação (Etapa 4 - KDD)
Responsável por engenharia de variáveis, reamostragens (horária -> diária e mensal),
cálculo de amplitude térmica, dias secos, anomalias climatológicas e persistência.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def get_southern_hemisphere_season(month: int) -> str:
    """
    Retorna a estação do ano meteorológica para o Hemisfério Sul.
    """
    if month in (12, 1, 2):
        return "Verão"
    elif month in (3, 4, 5):
        return "Outono"
    elif month in (6, 7, 8):
        return "Inverno"
    else:
        return "Primavera"


def build_daily_features(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Reamostra a série horária para frequência diária com agregações meteorológicas:
    - Médias, máximas, mínimas de temperatura e amplitude térmica
    - Acumulado de chuva e detecção de dias secos consecutivos
    - Vento máximo e médias de pressão e umidade
    """
    daily = pd.DataFrame(index=df_hourly.resample("D").asfreq().index)

    # Temperatura
    daily["temp_mean"] = df_hourly["temperature_2m"].resample("D").mean()
    daily["temp_min"] = df_hourly["temperature_2m"].resample("D").min()
    daily["temp_max"] = df_hourly["temperature_2m"].resample("D").max()
    daily["thermal_amplitude"] = daily["temp_max"] - daily["temp_min"]

    # Umidade
    daily["humidity_mean"] = df_hourly["relative_humidity_2m"].resample("D").mean()
    daily["humidity_min"] = df_hourly["relative_humidity_2m"].resample("D").min()

    # Precipitação
    daily["precipitation_sum"] = df_hourly["precipitation"].resample("D").sum()
    daily["is_rainy_day"] = (daily["precipitation_sum"] >= 1.0).astype(int)

    # Vento
    daily["wind_speed_mean"] = df_hourly["wind_speed_10m"].resample("D").mean()
    daily["wind_speed_max"] = df_hourly["wind_speed_10m"].resample("D").max()

    # Pressão e Radiação
    daily["surface_pressure_mean"] = df_hourly["surface_pressure"].resample("D").mean()
    daily["solar_radiation_sum"] = df_hourly["shortwave_radiation"].resample("D").sum()

    # Variáveis de calendário
    daily["year"] = daily.index.year
    daily["month"] = daily.index.month
    daily["day"] = daily.index.day
    daily["dayofweek"] = daily.index.dayofweek
    daily["season"] = daily["month"].apply(get_southern_hemisphere_season)

    # Dias consecutivos sem chuva (dry spells)
    is_dry = daily["is_rainy_day"] == 0
    dry_spell_counter = []
    current_count = 0
    for dry in is_dry:
        if dry:
            current_count += 1
        else:
            current_count = 0
        dry_spell_counter.append(current_count)
    daily["consecutive_dry_days"] = dry_spell_counter

    # Médias móveis (7 e 30 dias)
    daily["temp_mean_roll7"] = daily["temp_mean"].rolling(window=7, min_periods=1).mean()
    daily["temp_mean_roll30"] = daily["temp_mean"].rolling(window=30, min_periods=1).mean()
    daily["precip_sum_roll7"] = daily["precipitation_sum"].rolling(window=7, min_periods=1).sum()
    daily["precip_sum_roll30"] = daily["precipitation_sum"].rolling(window=30, min_periods=1).sum()

    # Lags (1 e 7 dias)
    daily["temp_mean_lag1"] = daily["temp_mean"].shift(1)
    daily["temp_mean_lag7"] = daily["temp_mean"].shift(7)

    return daily


def build_monthly_climatology(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os dados diários em mensais e calcula anomalias em relação à climatologia 2010-2024.
    """
    monthly = pd.DataFrame(index=df_daily.resample("MS").asfreq().index)

    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month
    monthly["season"] = monthly["month"].apply(get_southern_hemisphere_season)

    monthly["temp_mean"] = df_daily["temp_mean"].resample("MS").mean()
    monthly["temp_min"] = df_daily["temp_min"].resample("MS").min()
    monthly["temp_max"] = df_daily["temp_max"].resample("MS").max()
    monthly["thermal_amplitude_mean"] = df_daily["thermal_amplitude"].resample("MS").mean()

    monthly["precipitation_total"] = df_daily["precipitation_sum"].resample("MS").sum()
    monthly["rainy_days_count"] = df_daily["is_rainy_day"].resample("MS").sum()
    monthly["max_consecutive_dry_days"] = df_daily["consecutive_dry_days"].resample("MS").max()

    monthly["humidity_mean"] = df_daily["humidity_mean"].resample("MS").mean()
    monthly["wind_speed_mean"] = df_daily["wind_speed_mean"].resample("MS").mean()

    # Climatologia base do período (médias normais de cada um dos 12 meses)
    climatology_temp = monthly.groupby("month")["temp_mean"].transform("mean")
    climatology_precip = monthly.groupby("month")["precipitation_total"].transform("mean")

    monthly["temp_climatology_mean"] = climatology_temp
    monthly["precip_climatology_mean"] = climatology_precip

    # Anomalias em relação à climatologia
    monthly["temp_anomaly"] = monthly["temp_mean"] - monthly["temp_climatology_mean"]
    monthly["precip_anomaly"] = monthly["precipitation_total"] - monthly["precip_climatology_mean"]

    return monthly


def run_transformation_pipeline(
    processed_parquet: str = "data/processed/bauru_weather_processed.parquet",
    output_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa a transformação gerando datasets diário e mensal enriquecidos.
    """
    print(f"1. Lendo dataset horários limpo de: {processed_parquet}")
    df_hourly = pd.read_parquet(processed_parquet)

    print("2. Construindo agregados e features diárias...")
    df_daily = build_daily_features(df_hourly)

    print("3. Construindo agregados mensais e cálculo de anomalias climatológicas...")
    df_monthly = build_monthly_climatology(df_daily)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    daily_parquet = out_path / "bauru_weather_daily.parquet"
    daily_csv = out_path / "bauru_weather_daily.csv"
    monthly_parquet = out_path / "bauru_weather_monthly.parquet"
    monthly_csv = out_path / "bauru_weather_monthly.csv"

    df_daily.to_parquet(daily_parquet)
    df_daily.to_csv(daily_csv)
    df_monthly.to_parquet(monthly_parquet)
    df_monthly.to_csv(monthly_csv)

    print(f"-> Dataset diário salvo em: {daily_parquet} ({df_daily.shape[0]} registros x {df_daily.shape[1]} colunas)")
    print(f"-> Dataset mensal salvo em: {monthly_parquet} ({df_monthly.shape[0]} registros x {df_monthly.shape[1]} colunas)")

    return df_daily, df_monthly


if __name__ == "__main__":
    run_transformation_pipeline()
