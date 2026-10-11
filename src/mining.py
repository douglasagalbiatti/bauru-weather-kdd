"""
Módulo de Mineração de Dados e Inferência Estatística (Etapa 5 - KDD)
Responsável por testes de estacionariedade (ADF/KPSS), testes de tendência de
Mann-Kendall com Sen's slope, testes de hipóteses interdecadais e decomposição STL.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple
import warnings

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss


def test_stationarity(series: pd.Series, name: str) -> Dict[str, Any]:
    """
    Executa testes ADF e KPSS em uma série temporal.
    """
    clean_series = series.dropna()
    adf_res = adfuller(clean_series, autolag="AIC")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_res = kpss(clean_series, regression="c", nlags="auto")

    return {
        "variable": name,
        "adf": {
            "test_statistic": float(adf_res[0]),
            "p_value": float(adf_res[1]),
            "is_stationary_p05": bool(adf_res[1] < 0.05),
        },
        "kpss": {
            "test_statistic": float(kpss_res[0]),
            "p_value": float(kpss_res[1]),
            "is_stationary_p05": bool(kpss_res[1] >= 0.05),
        },
    }


def test_mann_kendall_trend(series: pd.Series, name: str) -> Dict[str, Any]:
    """
    Executa o teste de tendência não paramétrico de Mann-Kendall e estimação de Sen's slope.
    """
    clean_series = series.dropna()
    res = mk.original_test(clean_series)
    return {
        "variable": name,
        "trend": str(res.trend),
        "hypothesis_p_value": float(res.p),
        "z_score": float(res.z),
        "tau": float(res.Tau),
        "sens_slope": float(res.slope),
        "has_statistically_significant_trend": bool(res.h),
    }


def compare_subperiods(df_daily: pd.DataFrame) -> Dict[str, Any]:
    """
    Compara o período 2010-2016 contra 2017-2024 para temperatura e estiagem,
    verificando hipóteses de aquecimento ou aumento de extremos.
    """
    period_1 = df_daily[df_daily["year"] <= 2016]
    period_2 = df_daily[df_daily["year"] > 2016]

    temp_p1 = period_1["temp_mean"].dropna()
    temp_p2 = period_2["temp_mean"].dropna()

    ttest_temp = stats.ttest_ind(temp_p1, temp_p2, equal_var=False)
    mwu_temp = stats.mannwhitneyu(temp_p1, temp_p2, alternative="two-sided")

    # Comparação de dias consecutivos sem chuva
    dry_p1 = period_1["consecutive_dry_days"].dropna()
    dry_p2 = period_2["consecutive_dry_days"].dropna()
    mwu_dry = stats.mannwhitneyu(dry_p1, dry_p2, alternative="two-sided")

    return {
        "period_1_years": "2010-2016",
        "period_2_years": "2017-2024",
        "temperature_mean_c": {
            "period_1": float(temp_p1.mean()),
            "period_2": float(temp_p2.mean()),
            "difference": float(temp_p2.mean() - temp_p1.mean()),
            "ttest_p_value": float(ttest_temp.pvalue),
            "mann_whitney_p_value": float(mwu_temp.pvalue),
            "statistically_different": bool(ttest_temp.pvalue < 0.05),
        },
        "consecutive_dry_days_max": {
            "period_1_max": int(period_1["consecutive_dry_days"].max()),
            "period_2_max": int(period_2["consecutive_dry_days"].max()),
            "mann_whitney_p_value": float(mwu_dry.pvalue),
        },
    }


def compute_stl_decomposition(
    series: pd.Series, period: int = 12
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula a decomposição STL da série mensal.
    """
    clean_series = series.dropna()
    stl = STL(clean_series, period=period, robust=True)
    res = stl.fit()
    return res.trend, res.seasonal, res.resid


def run_mining_pipeline(
    daily_parquet: str = "data/processed/bauru_weather_daily.parquet",
    monthly_parquet: str = "data/processed/bauru_weather_monthly.parquet",
    output_dir: str = "reports/tables",
) -> Dict[str, Any]:
    """
    Executa a bateria de mineração estatística e persiste os resultados.
    """
    print(f"1. Lendo datasets processados...")
    df_daily = pd.read_parquet(daily_parquet)
    df_monthly = pd.read_parquet(monthly_parquet)

    print("2. Calculando testes de estacionariedade (ADF & KPSS)...")
    stationarity_results = [
        test_stationarity(df_daily["temp_mean"], "Temperatura Média Diária"),
        test_stationarity(df_daily["precipitation_sum"], "Precipitação Diária"),
        test_monthly_temp := test_stationarity(df_monthly["temp_mean"], "Temperatura Média Mensal"),
        test_stationarity(df_monthly["precipitation_total"], "Precipitação Mensal"),
    ]

    print("3. Executando testes de tendência de Mann-Kendall...")
    trend_results = [
        test_mann_kendall_trend(df_monthly["temp_mean"], "Temperatura Média Mensal"),
        test_mann_kendall_trend(df_monthly["temp_max"], "Temperatura Máxima Mensal"),
        test_mann_kendall_trend(df_monthly["temp_min"], "Temperatura Mínima Mensal"),
        test_mann_kendall_trend(df_monthly["precipitation_total"], "Precipitação Total Mensal"),
        test_mann_kendall_trend(df_monthly["max_consecutive_dry_days"], "Máximo de Dias Secos Consecutivos"),
    ]

    print("4. Realizando inferência comparativa entre subperíodos (2010-2016 vs 2017-2024)...")
    subperiod_comparison = compare_subperiods(df_daily)

    full_report = {
        "stationarity_tests": stationarity_results,
        "mann_kendall_trends": trend_results,
        "subperiod_comparison": subperiod_comparison,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_file = out_path / "statistical_tests.json"
    trends_csv = out_path / "trend_analysis_summary.csv"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    df_trends = pd.DataFrame(trend_results)
    df_trends.to_csv(trends_csv, index=False)

    print(f"-> Relatório estatístico salvo em: {summary_file}")
    print(f"-> Resumo de tendências salvo em: {trends_csv}")
    return full_report


if __name__ == "__main__":
    run_mining_pipeline()
