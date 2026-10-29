"""
Módulo de Mineração de Dados e Inferência Estatística (Etapa 5 - KDD)
Responsável por testes de estacionariedade (ADF/KPSS), testes de tendência de
Mann-Kendall (sazonal para séries mensais) com Sen's slope, comparação entre
subperíodos e decomposição STL.

Revisão de 29/10/2026:
- Toda a inferência usa a janela fixa do estudo (2010–2024, anos completos), para que
  os resultados não mudem a cada atualização diária da base.
- Séries mensais usam o Mann-Kendall SAZONAL (compara cada mês com o mesmo mês dos
  outros anos); o teste original assume ausência de sazonalidade.
- Estiagens são comparadas por EVENTO, não pelo contador diário (não independente).
- Temperaturas são comparadas pelas médias ANUAIS, não por dias (autocorrelacionados).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import warnings

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.transformation import extract_dry_spell_events  # noqa: E402

STUDY_YEARS = (2010, 2024)
PERIOD_1 = (2010, 2016)
PERIOD_2 = (2017, 2024)
ALPHA = 0.05


def _between(df: pd.DataFrame, years: Tuple[int, int]) -> pd.DataFrame:
    return df[(df.index.year >= years[0]) & (df.index.year <= years[1])]


def test_stationarity(series: pd.Series, name: str) -> Dict[str, Any]:
    """
    Executa testes ADF e KPSS em uma série temporal.
    ADF: H0 = raiz unitária (não estacionária). KPSS: H0 = estacionária.
    """
    clean_series = series.dropna()
    adf_res = adfuller(clean_series, autolag="AIC", result_object=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_res = kpss(clean_series, regression="c", nlags="auto")

    return {
        "variable": name,
        "adf": {
            "test_statistic": float(adf_res[0]),
            "p_value": float(adf_res[1]),
            "is_stationary_p05": bool(adf_res[1] < ALPHA),
        },
        "kpss": {
            "test_statistic": float(kpss_res[0]),
            "p_value": float(kpss_res[1]),
            "p_value_is_lower_bound": bool(kpss_res[1] >= 0.1),
            "is_stationary_p05": bool(kpss_res[1] >= ALPHA),
        },
    }


def test_mann_kendall_trend(series: pd.Series, name: str, seasonal_period: int = 0) -> Dict[str, Any]:
    """
    Teste de tendência de Mann-Kendall com Sen's slope.
    seasonal_period=12 aplica a versão sazonal (indicada para séries mensais).
    """
    clean_series = series.dropna()
    if seasonal_period:
        res = mk.seasonal_test(clean_series.values, period=seasonal_period, alpha=ALPHA)
        test_name = f"Mann-Kendall sazonal (período {seasonal_period})"
    else:
        res = mk.original_test(clean_series.values, alpha=ALPHA)
        test_name = "Mann-Kendall original"
    return {
        "variable": name,
        "test": test_name,
        "n": int(len(clean_series)),
        "trend": str(res.trend),
        "hypothesis_p_value": float(res.p),
        "z_score": float(res.z),
        "tau": float(res.Tau),
        "sens_slope": float(res.slope),
        "has_statistically_significant_trend": bool(res.h),
    }


def compare_dry_spells(df_daily: pd.DataFrame) -> Dict[str, Any]:
    """
    Compara as estiagens dos dois subperíodos tratando cada evento como uma observação.
    """
    events = extract_dry_spell_events(df_daily)
    events = events[(events["year"] >= STUDY_YEARS[0]) & (events["year"] <= STUDY_YEARS[1])]
    e1 = events[(events["year"] >= PERIOD_1[0]) & (events["year"] <= PERIOD_1[1])]["duration_days"]
    e2 = events[(events["year"] >= PERIOD_2[0]) & (events["year"] <= PERIOD_2[1])]["duration_days"]

    mwu = stats.mannwhitneyu(e1, e2, alternative="two-sided")

    def resumo(e: pd.Series, anos: Tuple[int, int]) -> Dict[str, Any]:
        n_anos = anos[1] - anos[0] + 1
        return {
            "years": f"{anos[0]}-{anos[1]}",
            "n_events": int(len(e)),
            "events_per_year": round(len(e) / n_anos, 2),
            "mean_duration_days": float(e.mean()),
            "median_duration_days": float(e.median()),
            "max_duration_days": int(e.max()),
            "n_events_10_days_or_more": int((e >= 10).sum()),
            "n_events_30_days_or_more": int((e >= 30).sum()),
        }

    return {
        "unit": "evento de estiagem (sequência de dias com chuva < 1 mm)",
        "period_1": resumo(e1, PERIOD_1),
        "period_2": resumo(e2, PERIOD_2),
        "mann_whitney_duration_p_value": float(mwu.pvalue),
        "statistically_different": bool(mwu.pvalue < ALPHA),
    }


def compare_subperiods(df_daily: pd.DataFrame) -> Dict[str, Any]:
    """
    Compara 2010–2016 com 2017–2024.
    Temperatura: médias anuais (Welch e Mann-Whitney). Estiagem: por evento.
    """
    study = _between(df_daily, STUDY_YEARS)
    p1 = _between(study, PERIOD_1)
    p2 = _between(study, PERIOD_2)

    annual_temp = study["temp_mean"].groupby(study.index.year).mean()
    t1 = annual_temp.loc[PERIOD_1[0]:PERIOD_1[1]]
    t2 = annual_temp.loc[PERIOD_2[0]:PERIOD_2[1]]
    ttest_temp = stats.ttest_ind(t1, t2, equal_var=False)
    mwu_temp = stats.mannwhitneyu(t1, t2, alternative="two-sided")

    return {
        "period_1_years": f"{PERIOD_1[0]}-{PERIOD_1[1]}",
        "period_2_years": f"{PERIOD_2[0]}-{PERIOD_2[1]}",
        "temperature_mean_c": {
            "unit": "média anual",
            "period_1": float(p1["temp_mean"].mean()),
            "period_2": float(p2["temp_mean"].mean()),
            "difference": float(p2["temp_mean"].mean() - p1["temp_mean"].mean()),
            "ttest_p_value": float(ttest_temp.pvalue),
            "mann_whitney_p_value": float(mwu_temp.pvalue),
            "statistically_different": bool(ttest_temp.pvalue < ALPHA),
        },
        "consecutive_dry_days_max": {
            "period_1_max": int(p1["consecutive_dry_days"].max()),
            "period_2_max": int(p2["consecutive_dry_days"].max()),
            "period_2_max_date": str(p2["consecutive_dry_days"].idxmax().date()),
        },
        "dry_spell_events": compare_dry_spells(study),
    }


def climate_summary(df_daily: pd.DataFrame, df_monthly: pd.DataFrame) -> Dict[str, Any]:
    """
    Números descritivos usados no relatório e no app (janela do estudo).
    """
    study = _between(df_daily, STUDY_YEARS)
    months = df_monthly[(df_monthly["year"] >= STUDY_YEARS[0]) & (df_monthly["year"] <= STUDY_YEARS[1])]
    temp_month = months.groupby("month")["temp_mean"].mean()
    precip_month = months.groupby("month")["precipitation_total"].mean()
    amp_month = study.groupby(study.index.month)["thermal_amplitude"].mean()

    corr_pairs = {
        "temp_max_x_solar_radiation_sum": ("temp_max", "solar_radiation_sum"),
        "precipitation_sum_x_thermal_amplitude": ("precipitation_sum", "thermal_amplitude"),
        "temp_mean_x_surface_pressure_mean": ("temp_mean", "surface_pressure_mean"),
        "humidity_mean_x_thermal_amplitude": ("humidity_mean", "thermal_amplitude"),
    }
    annual_max_dry = study["consecutive_dry_days"].groupby(study.index.year).max()

    return {
        "study_years": f"{STUDY_YEARS[0]}-{STUDY_YEARS[1]}",
        "n_days": int(len(study)),
        "n_months": int(len(months)),
        "temp_mean_c": float(study["temp_mean"].mean()),
        "annual_precip_mean_mm": float(study["precipitation_sum"].groupby(study.index.year).sum().mean()),
        "monthly_temp_normals_c": {int(k): round(float(v), 2) for k, v in temp_month.items()},
        "monthly_precip_normals_mm": {int(k): round(float(v), 1) for k, v in precip_month.items()},
        "monthly_amplitude_c": {int(k): round(float(v), 2) for k, v in amp_month.items()},
        "temp_max_record": {"value_c": float(study["temp_max"].max()), "date": str(study["temp_max"].idxmax().date())},
        "temp_min_record": {"value_c": float(study["temp_min"].min()), "date": str(study["temp_min"].idxmin().date())},
        "annual_max_dry_spell_days": {int(k): int(v) for k, v in annual_max_dry.items()},
        "pearson_correlations": {
            k: round(float(study[a].corr(study[b])), 3) for k, (a, b) in corr_pairs.items()
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
    print("1. Lendo datasets processados...")
    df_daily = _between(pd.read_parquet(daily_parquet), STUDY_YEARS)
    df_monthly = _between(pd.read_parquet(monthly_parquet), STUDY_YEARS)
    print(f"   Janela do estudo: {STUDY_YEARS[0]}–{STUDY_YEARS[1]} ({len(df_daily)} dias, {len(df_monthly)} meses)")

    print("2. Calculando testes de estacionariedade (ADF & KPSS)...")
    stationarity_results = [
        test_stationarity(df_daily["temp_mean"], "Temperatura Média Diária"),
        test_stationarity(df_daily["precipitation_sum"], "Precipitação Diária"),
        test_stationarity(df_monthly["temp_mean"], "Temperatura Média Mensal"),
        test_stationarity(df_monthly["precipitation_total"], "Precipitação Mensal"),
    ]

    print("3. Executando testes de tendência de Mann-Kendall (sazonal para séries mensais)...")
    annual_max_dry = df_daily["consecutive_dry_days"].groupby(df_daily.index.year).max()
    trend_results = [
        test_mann_kendall_trend(df_monthly["temp_mean"], "Temperatura Média Mensal", seasonal_period=12),
        test_mann_kendall_trend(df_monthly["temp_max"], "Temperatura Máxima Mensal", seasonal_period=12),
        test_mann_kendall_trend(df_monthly["temp_min"], "Temperatura Mínima Mensal", seasonal_period=12),
        test_mann_kendall_trend(df_monthly["precipitation_total"], "Precipitação Total Mensal", seasonal_period=12),
        test_mann_kendall_trend(annual_max_dry, "Maior Estiagem de Cada Ano"),
    ]

    print("4. Comparando subperíodos (2010–2016 vs 2017–2024): médias anuais e eventos de estiagem...")
    subperiod_comparison = compare_subperiods(df_daily)

    print("5. Resumo climatológico e correlações...")
    summary = climate_summary(df_daily, df_monthly)

    full_report = {
        "study_window": f"{STUDY_YEARS[0]}-{STUDY_YEARS[1]}",
        "alpha": ALPHA,
        "stationarity_tests": stationarity_results,
        "mann_kendall_trends": trend_results,
        "subperiod_comparison": subperiod_comparison,
        "climate_summary": summary,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_file = out_path / "statistical_tests.json"
    trends_csv = out_path / "trend_analysis_summary.csv"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    pd.DataFrame(trend_results).to_csv(trends_csv, index=False)

    ev = subperiod_comparison["dry_spell_events"]
    print(f"   Estiagens por evento: p = {ev['mann_whitney_duration_p_value']:.3f} "
          f"({'diferença significativa' if ev['statistically_different'] else 'sem diferença significativa'})")
    print(f"-> Relatório estatístico salvo em: {summary_file}")
    print(f"-> Resumo de tendências salvo em: {trends_csv}")
    return full_report


if __name__ == "__main__":
    run_mining_pipeline()
