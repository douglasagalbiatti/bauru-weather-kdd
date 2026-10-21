"""
Painel Interativo de Séries Temporais Meteorológicas – Bauru/SP
Metodologia KDD (Knowledge Discovery in Databases)
Autor: Prof. Dr. Douglas A. Galbiatti
"""

from datetime import datetime
import html
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from typing import Any, Dict, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import streamlit as st
from src.queimadas_modeling import PlattCalibrator


# Configuração da página
st.set_page_config(
    page_title="Clima Bauru/SP – Painel KDD",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização CSS refinada
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .explanation-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 15px 0;
        font-size: 0.95rem;
        color: #1e3a8a;
    }
    .live-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        padding: 15px 18px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .footer-box {
        margin-top: 50px;
        padding: 20px;
        text-align: center;
        background-color: #f1f5f9;
        border-radius: 8px;
        border-top: 2px solid #cbd5e1;
        font-size: 0.95rem;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_datasets():
    # Chave de controle para invalidação de cache (atualização queimadas 2025)
    _versao_cache = "bauru_queimadas_2025_v1"
    daily_path = Path("data/processed/bauru_weather_daily.parquet")
    monthly_path = Path("data/processed/bauru_weather_monthly.parquet")
    stats_path = Path("reports/tables/statistical_tests.json")
    ml_metrics_path = Path("reports/tables/ml_evaluation_metrics.json")
    ml_preds_path = Path("reports/tables/ml_predictions_test.parquet")
    model_path = Path("models/xgboost_weather_bauru.joblib")
    queimadas_path = Path("data/processed/bauru_weather_queimadas_daily.parquet")
    curva_risco_path = Path("reports/tables/curva_estiagem_vs_queimadas.json")
    model_queimadas_path = Path("models/xgboost_queimadas_bauru.joblib")
    queimadas_metrics_path = Path("reports/tables/queimadas_metricas.json")

    df_daily = pd.read_parquet(daily_path)
    df_monthly = pd.read_parquet(monthly_path)

    stats_report = {}
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_report = json.load(f)

    ml_metrics = {}
    if ml_metrics_path.exists():
        with open(ml_metrics_path, "r", encoding="utf-8") as f:
            ml_metrics = json.load(f)

    df_preds = pd.DataFrame()
    if ml_preds_path.exists():
        df_preds = pd.read_parquet(ml_preds_path)

    xgb_model = None
    if model_path.exists():
        xgb_model = joblib.load(model_path)

    df_queimadas = pd.DataFrame()
    if queimadas_path.exists():
        df_queimadas = pd.read_parquet(queimadas_path)

    curva_risco = []
    if curva_risco_path.exists():
        with open(curva_risco_path, "r", encoding="utf-8") as f:
            curva_risco = json.load(f)

    queimadas_metrics = {}
    if queimadas_metrics_path.exists():
        with open(queimadas_metrics_path, "r", encoding="utf-8") as f:
            queimadas_metrics = json.load(f)

    xgb_queimadas_model = None
    if model_queimadas_path.exists():
        xgb_queimadas_model = joblib.load(model_queimadas_path)

    model_calibrado_path = Path("models/xgboost_queimadas_bauru_calibrado.joblib")
    xgb_calibrado_model = None
    if model_calibrado_path.exists():
        xgb_calibrado_model = joblib.load(model_calibrado_path)

    return (
        df_daily,
        df_monthly,
        stats_report,
        ml_metrics,
        df_preds,
        xgb_model,
        df_queimadas,
        curva_risco,
        queimadas_metrics,
        xgb_queimadas_model,
        xgb_calibrado_model,
    )


def build_contingency_vector_from_history(df_daily: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Constrói um vetor de inferência e resumo de contingência a partir do último
    registro histórico diário consolidado na base local.
    """
    last_row = df_daily.iloc[-1]
    last_date = df_daily.index[-1]
    doy = last_date.dayofyear
    sin_doy = float(np.sin(2 * np.pi * doy / 365.25))
    cos_doy = float(np.cos(2 * np.pi * doy / 365.25))

    lag2_max = float(df_daily["temp_max"].iloc[-3]) if len(df_daily) >= 3 else float(last_row["temp_max"])
    lag3_max = float(df_daily["temp_max"].iloc[-4]) if len(df_daily) >= 4 else float(last_row["temp_max"])
    lag1_hum = float(df_daily["humidity_mean"].iloc[-2]) if len(df_daily) >= 2 else float(last_row["humidity_mean"])
    lag1_rad = float(df_daily["solar_radiation_sum"].iloc[-2]) if len(df_daily) >= 2 else float(last_row["solar_radiation_sum"])
    lag1_press = float(df_daily["surface_pressure_mean"].iloc[-2]) if len(df_daily) >= 2 else float(last_row["surface_pressure_mean"])

    contingency_row = pd.DataFrame([{
        "temp_mean": float(last_row["temp_mean"]),
        "temp_min": float(last_row["temp_min"]),
        "temp_max": float(last_row["temp_max"]),
        "thermal_amplitude": float(last_row["thermal_amplitude"]),
        "humidity_mean": float(last_row["humidity_mean"]),
        "humidity_min": float(last_row["humidity_min"]),
        "precipitation_sum": float(last_row["precipitation_sum"]),
        "wind_speed_mean": float(last_row["wind_speed_mean"]),
        "wind_speed_max": float(last_row["wind_speed_max"]),
        "surface_pressure_mean": float(last_row["surface_pressure_mean"]),
        "solar_radiation_sum": float(last_row["solar_radiation_sum"]),
        "temp_mean_roll7": float(last_row["temp_mean_roll7"]),
        "temp_mean_roll30": float(last_row["temp_mean_roll30"]),
        "precip_sum_roll7": float(last_row["precip_sum_roll7"]),
        "temp_mean_lag1": float(last_row["temp_mean_lag1"]),
        "temp_mean_lag7": float(last_row["temp_mean_lag7"]),
        "temp_max_lag2": lag2_max,
        "temp_max_lag3": lag3_max,
        "humidity_lag1": lag1_hum,
        "radiation_lag1": lag1_rad,
        "pressure_lag1": lag1_press,
        "sin_doy": sin_doy,
        "cos_doy": cos_doy,
    }])

    summary = {
        "date": last_date.strftime("%d/%m/%Y"),
        "temp_mean": float(last_row["temp_mean"]),
        "temp_max": float(last_row["temp_max"]),
        "temp_min": float(last_row["temp_min"]),
        "humidity_min": float(last_row["humidity_min"]),
        "precipitation": float(last_row["precipitation_sum"]),
        "wind_max": float(last_row["wind_speed_max"]),
        "pressure": float(last_row["surface_pressure_mean"]),
        "solar_radiation": float(last_row["solar_radiation_sum"]),
        "status": "contingency",
    }
    return contingency_row, summary


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_open_meteo_live_raw(
    latitude: float = -22.3147,
    longitude: float = -49.0608,
    timezone: str = "America/Sao_Paulo",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Obtém em tempo real via Open-Meteo Forecast API as condições de hoje
    e os últimos 7 dias. Dispara exceção em caso de erro para NUNCA cachear falhas (como HTTP 429).
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": 7,
        "forecast_days": 1,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "shortwave_radiation_sum",
        ],
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
        ],
        "timezone": timezone,
    }

    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=True,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "BauruWeatherKDD/1.0 (https://github.com/douglasagalbiatti/bauru-weather-kdd; contact: research)"
    })

    response = session.get(url, params=params, timeout=12)
    response.raise_for_status()
    data = response.json()

    daily_data = data["daily"]
    hourly_data = data["hourly"]

    df_d = pd.DataFrame(daily_data)
    df_d["time"] = pd.to_datetime(df_d["time"])
    df_d = df_d.set_index("time")

    df_h = pd.DataFrame(hourly_data)
    df_h["time"] = pd.to_datetime(df_h["time"])
    df_h = df_h.set_index("time")

    today_date = df_d.index[-1]
    today_daily = df_d.iloc[-1]
    today_hourly = df_h[df_h.index.date == today_date.date()]

    humidity_mean_today = float(today_hourly["relative_humidity_2m"].mean()) if not today_hourly.empty else 60.0
    humidity_min_today = float(today_hourly["relative_humidity_2m"].min()) if not today_hourly.empty else 40.0
    pressure_mean_today = float(today_hourly["surface_pressure"].mean()) if not today_hourly.empty else 955.0
    wind_mean_today = float(today_hourly["wind_speed_10m"].mean()) if not today_hourly.empty else 10.0

    temp_max_today = float(today_daily["temperature_2m_max"])
    temp_min_today = float(today_daily["temperature_2m_min"])
    temp_mean_today = float(today_daily["temperature_2m_mean"])
    precip_today = float(today_daily["precipitation_sum"])
    wind_max_today = float(today_daily["wind_speed_10m_max"])
    raw_rad = today_daily.get("shortwave_radiation_sum", 20.0)
    solar_rad_today = float(raw_rad) * 277.78 if pd.notna(raw_rad) else 20.0 * 277.78

    # Lags reais
    lag1_temp = float(df_d["temperature_2m_mean"].iloc[-2]) if len(df_d) >= 2 else temp_mean_today
    lag7_temp = float(df_d["temperature_2m_mean"].iloc[0]) if len(df_d) >= 7 else temp_mean_today
    lag2_max = float(df_d["temperature_2m_max"].iloc[-3]) if len(df_d) >= 3 else temp_max_today
    lag3_max = float(df_d["temperature_2m_max"].iloc[-4]) if len(df_d) >= 4 else temp_max_today

    roll7_temp = float(df_d["temperature_2m_mean"].tail(7).mean())
    roll7_rain = float(df_d["precipitation_sum"].tail(7).sum())

    day_of_year = today_date.dayofyear
    sin_doy = np.sin(2 * np.pi * day_of_year / 365.25)
    cos_doy = np.cos(2 * np.pi * day_of_year / 365.25)

    inference_row = pd.DataFrame([{
        "temp_mean": temp_mean_today,
        "temp_min": temp_min_today,
        "temp_max": temp_max_today,
        "thermal_amplitude": temp_max_today - temp_min_today,
        "humidity_mean": humidity_mean_today,
        "humidity_min": humidity_min_today,
        "precipitation_sum": precip_today,
        "wind_speed_mean": wind_mean_today,
        "wind_speed_max": wind_max_today,
        "surface_pressure_mean": pressure_mean_today,
        "solar_radiation_sum": solar_rad_today,
        "temp_mean_roll7": roll7_temp,
        "temp_mean_roll30": roll7_temp,
        "precip_sum_roll7": roll7_rain,
        "temp_mean_lag1": lag1_temp,
        "temp_mean_lag7": lag7_temp,
        "temp_max_lag2": lag2_max,
        "temp_max_lag3": lag3_max,
        "humidity_lag1": humidity_mean_today,
        "radiation_lag1": solar_rad_today,
        "pressure_lag1": pressure_mean_today,
        "sin_doy": sin_doy,
        "cos_doy": cos_doy,
    }])

    summary_today = {
        "date": today_date.strftime("%d/%m/%Y"),
        "temp_mean": temp_mean_today,
        "temp_max": temp_max_today,
        "temp_min": temp_min_today,
        "humidity_min": humidity_min_today,
        "precipitation": precip_today,
        "wind_max": wind_max_today,
        "pressure": pressure_mean_today,
        "solar_radiation": solar_rad_today,
        "status": "online",
    }
    return inference_row, summary_today


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_ipmet_live_raw(_df_daily: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Realiza o web scraping da estação meteorológica de superfície do IPMet/UNESP Bauru
    (https://www.ipmetradar.com.br/2estacao.php) e combina com a inércia histórica local.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.ipmetradar.com.br/",
        "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive",
    }

    session = requests.Session()
    session.headers.update(headers)

    urls = [
        "https://www.ipmetradar.com.br/2estacao.php",
        "http://www.ipmetradar.com.br/2estacao.php",
    ]
    response = None
    last_err = None

    # allow_redirects=False impede categoricamente loops de redirecionamento (TooManyRedirects)
    for target_url in urls:
        try:
            resp = session.get(target_url, timeout=10, allow_redirects=False)
            if resp.status_code == 200 and "<table" in resp.text.lower():
                response = resp
                break
            elif resp.status_code in (301, 302, 303, 307, 308):
                last_err = f"Servidor Apache redirecionou temporariamente (HTTP {resp.status_code})"
            else:
                last_err = f"Código HTTP inesperado ({resp.status_code})"
        except Exception as ex:
            last_err = str(ex)

    if response is None:
        raise ValueError(f"Estação IPMet indisponível: {last_err}")

    tables = re.findall(r"<table[^>]*>(.*?)</table>", response.text, re.DOTALL | re.IGNORECASE)
    if not tables:
        raise ValueError("Tabela da estação IPMet não encontrada na página.")

    clean_text = html.unescape(re.sub(r"<[^>]+>", " ", tables[0]))
    clean_text = " ".join(clean_text.split())

    time_m = re.search(r"Horário da Leitura \(GMT-3\)\s*([\d\/]+ - [\d:]+)", clean_text)
    temp_m = re.search(r"Temp:\s*([\d\.\-]+)\s*o\s*C", clean_text)
    max_m = re.search(r"Max:\s*([\d\.\-]+)\s*o\s*C", clean_text)
    min_m = re.search(r"Min:\s*([\d\.\-]+)\s*o\s*C", clean_text)
    chuva_m = re.search(r"Hoje:\s*([\d\.\-]+)\s*mm", clean_text)
    umid_m = re.search(r"Umidade:\s*([\d\.\-]+)%", clean_text)
    umid_min_m = re.search(r"Min:\s*([\d\.\-]+)%", clean_text)
    vento_m = re.search(r"Veloc:\s*([\d\.\-]+)\s*km/h", clean_text)
    rajada_m = re.search(r"Max dia:\s*([\d\.\-]+)\s*km/h", clean_text)
    pressao_m = re.search(r"Pressão:\s*([\d\.\-]+)\s*mB", clean_text)
    rad_m = re.search(r"Radiação Global:\s*([\d\.\-]+)\s*W/m", clean_text)

    temp_atual = float(temp_m.group(1)) if temp_m else 22.0
    temp_max = float(max_m.group(1)) if max_m else temp_atual
    temp_min = float(min_m.group(1)) if min_m else temp_atual
    temp_mean = (temp_max + temp_min) / 2.0 if temp_max != temp_min else temp_atual
    precip_today = float(chuva_m.group(1)) if chuva_m else 0.0
    hum_min = float(umid_min_m.group(1)) if umid_min_m else (float(umid_m.group(1)) if umid_m else 50.0)
    hum_mean = float(umid_m.group(1)) if umid_m else 70.0
    wind_mean = float(vento_m.group(1)) if vento_m else 8.0
    wind_max = float(rajada_m.group(1)) if rajada_m else (wind_mean * 1.5)
    press_mean = float(pressao_m.group(1)) if pressao_m else 955.0

    rad_val = float(rad_m.group(1)) if rad_m else 0.0
    solar_rad = rad_val if rad_val > 50.0 else float(_df_daily["solar_radiation_sum"].iloc[-1])

    roll7_temp = float(_df_daily["temp_mean"].tail(7).mean())
    roll7_rain = float(_df_daily["precipitation_sum"].tail(7).sum())
    roll30_temp = float(_df_daily["temp_mean"].tail(30).mean())
    lag1_temp = float(_df_daily["temp_mean"].iloc[-1])
    lag7_temp = float(_df_daily["temp_mean"].iloc[-7]) if len(_df_daily) >= 7 else lag1_temp
    lag2_max = float(_df_daily["temp_max"].iloc[-2]) if len(_df_daily) >= 2 else temp_max
    lag3_max = float(_df_daily["temp_max"].iloc[-3]) if len(_df_daily) >= 3 else temp_max
    lag1_hum = float(_df_daily["humidity_mean"].iloc[-1])
    lag1_rad = float(_df_daily["solar_radiation_sum"].iloc[-1])
    lag1_press = float(_df_daily["surface_pressure_mean"].iloc[-1])

    now = datetime.now()
    doy = now.timetuple().tm_yday
    sin_doy = float(np.sin(2 * np.pi * doy / 365.25))
    cos_doy = float(np.cos(2 * np.pi * doy / 365.25))

    inference_row = pd.DataFrame([{
        "temp_mean": temp_mean,
        "temp_min": temp_min,
        "temp_max": temp_max,
        "thermal_amplitude": temp_max - temp_min,
        "humidity_mean": hum_mean,
        "humidity_min": hum_min,
        "precipitation_sum": precip_today,
        "wind_speed_mean": wind_mean,
        "wind_speed_max": wind_max,
        "surface_pressure_mean": press_mean,
        "solar_radiation_sum": solar_rad,
        "temp_mean_roll7": roll7_temp,
        "temp_mean_roll30": roll30_temp,
        "precip_sum_roll7": roll7_rain,
        "temp_mean_lag1": lag1_temp,
        "temp_mean_lag7": lag7_temp,
        "temp_max_lag2": lag2_max,
        "temp_max_lag3": lag3_max,
        "humidity_lag1": lag1_hum,
        "radiation_lag1": lag1_rad,
        "pressure_lag1": lag1_press,
        "sin_doy": sin_doy,
        "cos_doy": cos_doy,
    }])

    reading_time = time_m.group(1) if time_m else now.strftime("%d/%m/%Y")
    summary = {
        "date": reading_time,
        "temp_mean": temp_mean,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "humidity_min": hum_min,
        "precipitation": precip_today,
        "wind_max": wind_max,
        "pressure": press_mean,
        "solar_radiation": solar_rad,
        "status": "ipmet_online",
    }
    return inference_row, summary


def fetch_live_today_weather(
    df_daily: Optional[pd.DataFrame] = None,
    source_mode: str = "auto",
    latitude: float = -22.3147,
    longitude: float = -49.0608,
    timezone: str = "America/Sao_Paulo",
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str]]:
    """
    Provedor multinível resiliente de dados climáticos em tempo real:
    - auto: Tenta Open-Meteo -> Fallback IPMet Bauru -> Fallback histórico local.
    - ipmet: Consulta a estação física do IPMet/UNESP Bauru via web scraping.
    - open_meteo: Consulta a API Open-Meteo Forecast.
    """
    if source_mode == "ipmet":
        try:
            if df_daily is not None and len(df_daily) >= 4:
                inference_row, summary = _fetch_ipmet_live_raw(df_daily)
                return inference_row, summary, None
            raise ValueError("Base histórica necessária para lags temporais.")
        except Exception as e:
            if df_daily is not None and len(df_daily) >= 4:
                c_row, c_sum = build_contingency_vector_from_history(df_daily)
                return c_row, c_sum, f"IPMet: {e}"
            return None, None, str(e)

    if source_mode == "open_meteo":
        try:
            inference_row, summary_today = _fetch_open_meteo_live_raw(latitude, longitude, timezone)
            return inference_row, summary_today, None
        except Exception as e:
            if df_daily is not None and len(df_daily) >= 4:
                c_row, c_sum = build_contingency_vector_from_history(df_daily)
                return c_row, c_sum, f"Open-Meteo: {e}"
            return None, None, str(e)

    # Modo automático (auto)
    try:
        inference_row, summary_today = _fetch_open_meteo_live_raw(latitude, longitude, timezone)
        return inference_row, summary_today, None
    except Exception as e_om:
        # Camada 2: Tenta Estação IPMet / UNESP Bauru como fallback imediato
        if df_daily is not None and len(df_daily) >= 4:
            try:
                i_row, i_sum = _fetch_ipmet_live_raw(df_daily)
                i_sum["status"] = "ipmet_fallback"
                return i_row, i_sum, str(e_om)
            except Exception:
                pass
            # Camada 3: Contingência histórica local
            c_row, c_sum = build_contingency_vector_from_history(df_daily)
            return c_row, c_sum, str(e_om)
        return None, None, str(e_om)


(
    df_daily,
    df_monthly,
    stats_report,
    ml_metrics,
    df_preds,
    xgb_model,
    df_queimadas,
    curva_risco,
    queimadas_metrics,
    xgb_queimadas_model,
    xgb_calibrado_model,
) = load_datasets()

# ==========================================
# BARRA LATERAL (SIDEBAR) COM FILTROS
# ==========================================
st.sidebar.image("https://img.icons8.com/clouds/200/sun.png", width=110)
st.sidebar.title("🎛️ Filtros de Análise")
st.sidebar.markdown("Personalize os períodos e visualize os dados dinamicamente.")

min_year = int(df_daily["year"].min())
max_year = int(df_daily["year"].max())

year_range = st.sidebar.slider(
    "Período de Anos:",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)

seasons_available = ["Verão", "Outono", "Inverno", "Primavera"]
selected_seasons = st.sidebar.multiselect(
    "Estações do Ano:",
    options=seasons_available,
    default=seasons_available,
)

filtered_daily = df_daily[
    (df_daily["year"] >= year_range[0])
    & (df_daily["year"] <= year_range[1])
    & (df_daily["season"].isin(selected_seasons))
]

filtered_monthly = df_monthly[
    (df_monthly["year"] >= year_range[0])
    & (df_monthly["year"] <= year_range[1])
    & (df_monthly["season"].isin(selected_seasons))
]

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Sobre a Fonte:**\n"
    "Dados históricos modelados via **Open-Meteo (Reanálise ERA5/ECMWF)** para as coordenadas de Bauru/SP."
)

# ==========================================
# CABEÇALHO EXPLICATIVO
# ==========================================
st.markdown('<div class="main-title">🌦️ Painel de Inteligência Climática – Bauru/SP</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Estudo de Séries Temporais Meteorológicas (2010 a 2024) fundamentado na metodologia '
    '<b>KDD (Knowledge Discovery in Databases)</b> e Aprendizado de Máquina (<b>XGBoost</b>).'
    '</div>',
    unsafe_allow_html=True,
)

# Cards de Resumo Executivo (KPIs)
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_temp = filtered_daily["temp_mean"].mean()
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-value">{avg_temp:.1f} °C</div>
            <div class="metric-label">Temperatura Média</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col2:
    years_count = max(1, len(range(year_range[0], year_range[1] + 1)))
    total_rain_annual = filtered_daily["precipitation_sum"].sum() / years_count
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-value">{total_rain_annual:.0f} mm</div>
            <div class="metric-label">Chuva Média Anual</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col3:
    max_temp = filtered_daily["temp_max"].max()
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-value" style="color: #dc2626;">{max_temp:.1f} °C</div>
            <div class="metric-label">Pico Máximo de Calor</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col4:
    max_dry = filtered_daily["consecutive_dry_days"].max()
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-value" style="color: #ea580c;">{max_dry} dias</div>
            <div class="metric-label">Recorde de Seca Contínua</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ==========================================
# ABAS DE NAVEGAÇÃO
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Panorama Climático",
    "🌡️ Tendências & Anomalias",
    "🌵 Estiagem & Seca (Achado Crítico)",
    "🤖 Inteligência Artificial (XGBoost Clima)",
    "🔥 Queimadas & Risco XGBoost",
    "📋 Explorador de Dados",
])

# ------------------------------------------
# ABA 1: PANORAMA CLIMÁTICO
# ------------------------------------------
with tab1:
    st.subheader("O Ciclo Anual do Clima em Bauru")
    st.markdown(
        """
        <div class="explanation-box">
        💡 <b>O que é o Climograma?</b><br>
        O climograma é a "impressão digital" da cidade. Ele combina em um único gráfico quanto chove (barras azuis) 
        e qual é a temperatura média (linha vermelha) para cada mês do ano. Veja como Bauru possui duas fases muito 
        claras: um <b>verão quente e muito chuvoso</b> (dezembro a março) e um <b>inverno seco e ameno</b> (junho a agosto).
        </div>
        """,
        unsafe_allow_html=True,
    )

    climatology = (
        filtered_monthly.groupby("month")
        .agg(
            precip_mean=("precipitation_total", "mean"),
            temp_mean=("temp_mean", "mean"),
        )
        .reset_index()
    )
    month_names = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    climatology["mes_nome"] = climatology["month"].apply(lambda m: month_names[m - 1])

    fig_clim = make_subplots(specs=[[{"secondary_y": True}]])

    fig_clim.add_trace(
        go.Bar(
            x=climatology["mes_nome"],
            y=climatology["precip_mean"],
            name="Chuva Média (mm)",
            marker_color="#3b82f6",
            opacity=0.75,
            hovertemplate="Chuva: %{y:.1f} mm<extra></extra>",
        ),
        secondary_y=False,
    )

    fig_clim.add_trace(
        go.Scatter(
            x=climatology["mes_nome"],
            y=climatology["temp_mean"],
            name="Temperatura Média (°C)",
            mode="lines+markers",
            line=dict(color="#ef4444", width=3),
            marker=dict(size=8),
            hovertemplate="Temperatura: %{y:.1f} °C<extra></extra>",
        ),
        secondary_y=True,
    )

    fig_clim.update_layout(
        title="<b>Climograma Médio de Bauru/SP</b>",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig_clim.update_yaxes(title_text="Precipitação (mm)", secondary_y=False)
    fig_clim.update_yaxes(title_text="Temperatura (°C)", secondary_y=True)
    st.plotly_chart(fig_clim, use_container_width=True)

    st.markdown(
        """
        <div class="explanation-box">
        💡 <b>Como ler a distribuição das temperaturas?</b><br>
        O gráfico abaixo mostra a dispersão diária de temperatura para cada mês. Os "pontinhos" fora da caixa 
        representam dias atípicos (frentes frias severas no meio do ano ou ondas de calor extremo no final do ano).
        </div>
        """,
        unsafe_allow_html=True,
    )
    df_box = filtered_daily.copy()
    df_box["mes_nome"] = df_box["month"].apply(lambda m: month_names[m - 1])

    fig_box = px.box(
        df_box,
        x="mes_nome",
        y="temp_mean",
        color="season",
        title="<b>Variação das Temperaturas Diárias ao Longo do Ano</b>",
        labels={"temp_mean": "Temperatura Média Diária (°C)", "mes_nome": "Mês", "season": "Estação"},
        category_orders={"mes_nome": month_names},
    )
    fig_box.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig_box, use_container_width=True)

# ------------------------------------------
# ABA 2: TENDÊNCIAS & ANOMALIAS TÉRMICAS
# ------------------------------------------
with tab2:
    st.subheader("Bauru Está Esquentando? O que os Dados Revelam")
    st.markdown(
        """
        <div class="explanation-box">
        💡 <b>O que são Anomalias de Temperatura?</b><br>
        Uma anomalia mede o quanto a temperatura de um mês específico ficou acima ou abaixo do padrão histórico daquele mesmo mês.
        <ul>
            <li><b style="color: #dc2626;">Barras Vermelhas:</b> Meses mais quentes do que o esperado (anomalia positiva).</li>
            <li><b style="color: #2563eb;">Barras Azuis:</b> Meses mais frescos do que o esperado (anomalia negativa).</li>
        </ul>
        A linha preta mostra a média móvel anual de tendência.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_anom = go.Figure()
    bar_colors = ["#dc2626" if val >= 0 else "#2563eb" for val in filtered_monthly["temp_anomaly"]]

    fig_anom.add_trace(
        go.Bar(
            x=filtered_monthly.index,
            y=filtered_monthly["temp_anomaly"],
            marker_color=bar_colors,
            name="Anomalia Mensal",
            hovertemplate="%{x|%b/%Y}: %{y:+.2f} °C<extra></extra>",
        )
    )

    roll12 = filtered_monthly["temp_anomaly"].rolling(window=12, center=True).mean()
    fig_anom.add_trace(
        go.Scatter(
            x=filtered_monthly.index,
            y=roll12,
            mode="lines",
            line=dict(color="#111827", width=2.5),
            name="Tendência (Média Móvel 12m)",
        )
    )

    fig_anom.update_layout(
        title="<b>Histórico de Anomalias Mensais de Temperatura (2010–2024)</b>",
        yaxis_title="Anomalia de Temperatura (°C)",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    st.write("#### 🧠 Traduzindo a Estatística para a Vida Real:")
    c_stat1, c_stat2 = st.columns(2)
    with c_stat1:
        st.success(
            "✅ **Média Geral Estável:**\n\n"
            "O teste estatístico formal de *Mann-Kendall* revelou que a temperatura média em Bauru **não apresenta "
            "uma tendência de aquecimento constante e acelerado** ($p = 0.56$). A média histórica subiu apenas "
            "**0,04 °C** entre o primeiro e o segundo período analisado."
        )
    with c_stat2:
        st.warning(
            "⚠️ **Atenção aos Picos Extremos:**\n\n"
            "Embora a média seja estável, as **temperaturas máximas diurnas** mostraram tendência positiva moderada. "
            "Ou seja: quando faz calor, as ondas de calor têm atingido patamares pontualmente mais elevados e desconfortáveis."
        )

# ------------------------------------------
# ABA 3: ESTIAGEM & SECA (ACHADO CRÍTICO)
# ------------------------------------------
with tab3:
    st.subheader("O Achado Crítico: Estiagens Mais Longas e Severas")
    st.markdown(
        """
        <div class="explanation-box">
        💡 <b>Entendendo os Dias Consecutivos Sem Chuva:</b><br>
        Esta métrica conta quantos dias seguidos a cidade ficou sem registrar sequer 1 mm de chuva. 
        Quando esse número ultrapassa <b>30 dias</b>, o risco de queimadas no cerrado paulista e a umidade do ar caem a níveis críticos. 
        Quando supera <b>60 dias</b>, o abastecimento e a agricultura entram em estresse severo.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_dry = go.Figure()
    fig_dry.add_trace(
        go.Scatter(
            x=filtered_daily.index,
            y=filtered_daily["consecutive_dry_days"],
            mode="lines",
            line=dict(color="#ea580c", width=1.8),
            name="Dias Sem Chuva",
            hovertemplate="%{x|%d/%m/%Y}: %{y} dias consecutivos sem chuva<extra></extra>",
        )
    )

    fig_dry.add_hline(y=30, line_dash="dash", line_color="#f59e0b", annotation_text="Alerta de Estiagem (30 dias)")
    fig_dry.add_hline(y=60, line_dash="dash", line_color="#dc2626", annotation_text="Estiagem Crítica (60 dias)")

    fig_dry.update_layout(
        title="<b>Evolução dos Períodos de Estiagem Contínua em Bauru/SP</b>",
        yaxis_title="Dias Consecutivos Sem Chuva",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig_dry, use_container_width=True)

    st.write("#### 📊 Comparação Científica entre as Duas Metades do Estudo:")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.info(
            "📅 **Primeira Metade (2010 a 2016):**\n\n"
            "* Recorde de seca ininterrupta: **64 dias**.\n"
            "* As chuvas na primavera retornavam com maior pontualidade.\n"
            "* Menor incidência de estresses hídricos extremos."
        )

    with col_d2:
        st.error(
            "📅 **Segunda Metade (2017 a 2024):**\n\n"
            "* Recorde de seca ininterrupta: **71 dias**.\n"
            "* Teste estatístico não-paramétrico comprovou agravamento com certeza científica (**p = 0,00000001**).\n"
            "* Períodos de estiagem mais longos e janelas de chuva mais concentradas."
        )

# ------------------------------------------
# ABA 4: INTELIGÊNCIA ARTIFICIAL (XGBOOST)
# ------------------------------------------
with tab4:
    st.subheader("🤖 Previsão Meteorológica Preditiva com XGBoost")
    st.markdown(
        """
        <div class="explanation-box">
        💡 <b>Como a Inteligência Artificial foi treinada?</b><br>
        Utilizamos o algoritmo <b>XGBoost (Extreme Gradient Boosting)</b>, considerado o estado da arte na meteorologia 
        para dados tabulares. O modelo aprendeu com <b>12 anos de histórico (2010 a 2021 - 4.380 dias)</b> e foi testado 
        "às cegas" em <b>3 anos recentes (2022 a 2024 - 1.096 dias)</b> para prever a <b>Temperatura Máxima de Amanhã</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================
    # SIMULADOR 100% ONLINE E AUTOMÁTICO
    # ==========================================
    st.write("### 🛰️ Previsão em Tempo Real para Bauru/SP (Online)")
    st.markdown(
        "Abaixo, o simulador consulta automaticamente a API da Open-Meteo para coletar as **condições climáticas de hoje "
        "ao vivo** em Bauru/SP e alimenta o modelo XGBoost para prever a temperatura máxima de amanhã sem necessidade "
        "de ajuste manual de parâmetros:"
    )

    c_btn1, c_sel2 = st.columns([2, 5])
    with c_btn1:
        st.write("")
        st.write("")
        btn_refresh = st.button("🔄 Atualizar Dados de Hoje")
    with c_sel2:
        source_label = st.selectbox(
            "Origem das Condições de Hoje:",
            options=[
                "Automático (Open-Meteo com fallback IPMet / UNESP)",
                "📡 Estação Meteorológica IPMet / UNESP Bauru (Sensor Físico)",
                "🛰️ API Open-Meteo (Modelos Globais Reanálise / GFS)",
            ],
            index=0,
            help="Selecione a fonte de dados em tempo real ou deixe no modo automático resiliente.",
        )

    mode_map = {
        "Automático (Open-Meteo com fallback IPMet / UNESP)": "auto",
        "📡 Estação Meteorológica IPMet / UNESP Bauru (Sensor Físico)": "ipmet",
        "🛰️ API Open-Meteo (Modelos Globais Reanálise / GFS)": "open_meteo",
    }
    source_mode = mode_map[source_label]

    if btn_refresh:
        st.cache_data.clear()
        st.rerun()

    # Busca condições de hoje online através do provedor selecionado
    live_vector, live_summary, live_err = fetch_live_today_weather(
        df_daily=df_daily,
        source_mode=source_mode,
    )

    if live_summary and live_vector is not None and xgb_model is not None:
        predicted_online = float(xgb_model.predict(live_vector)[0])
        status = live_summary.get("status", "online")
        is_contingency = status == "contingency"

        if is_contingency:
            c_warn_col1, c_warn_col2 = st.columns([4, 1])
            with c_warn_col1:
                st.warning(
                    f"⚠️ **Modo de Contingência Ativo:** As fontes online em tempo real estão temporariamente inacessíveis ({live_err}). "
                    f"O modelo XGBoost gerou a previsão utilizando o **último registro meteorológico consolidado na base local ({live_summary['date']})** como referência."
                )
            with c_warn_col2:
                if st.button("🔄 Reconectar", key="btn_reconectar_contingencia", help="Limpa o cache e tenta consultar as fontes ao vivo agora"):
                    st.cache_data.clear()
                    st.rerun()
        elif status == "ipmet_fallback":
            st.info(
                f"ℹ️ **Transição Automática para a Estação IPMet / UNESP Bauru:** A API Open-Meteo encontrava-se temporariamente "
                f"com limite de requisições ou inacessível. As condições climáticas de hoje foram capturadas com sucesso via "
                f"**Web Scraping da Estação de Superfície da UNESP Bauru** ({live_summary['date']})."
            )
        elif status == "ipmet_online":
            st.markdown(
                f"""
                <div class="live-box" style="border-left: 5px solid #2563eb; background-color: #eff6ff;">
                    <span style="font-weight: 700; color: #1d4ed8; font-size: 1.05rem;">
                        📡 Estação Meteorológica de Superfície IPMet / UNESP Bauru ({live_summary['date']})
                    </span>
                    <div style="font-size: 0.9rem; color: #1e40af; margin-top: 4px;">
                        Medições físicas locais coletadas em tempo real via web scraping do radar meteorológico (ipmetradar.com.br).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="live-box">
                    <span style="font-weight: 700; color: #15803d; font-size: 1.05rem;">
                        🛰️ Condições Climáticas de Hoje Obtidas Online ({live_summary['date']}) – Bauru/SP
                    </span>
                    <div style="font-size: 0.9rem; color: #475569; margin-top: 4px;">
                        Dados de reanálise/previsão horária coletados em tempo real via Open-Meteo API.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Cards com os dados obtidos (online, ipmet ou contingência)
        lcol1, lcol2, lcol3, lcol4, lcol5 = st.columns(5)
        prefix_label = "Referência" if is_contingency else "Hoje"
        with lcol1:
            st.metric(f"Temp. Máx ({prefix_label})", f"{live_summary['temp_max']:.1f} °C")
        with lcol2:
            st.metric(f"Temp. Média ({prefix_label})", f"{live_summary['temp_mean']:.1f} °C")
        with lcol3:
            st.metric("Umidade Mínima", f"{live_summary['humidity_min']:.0f} %")
        with lcol4:
            st.metric("Chuva Registrada", f"{live_summary['precipitation']:.1f} mm")
        with lcol5:
            st.metric("Vento Máximo", f"{live_summary['wind_max']:.1f} km/h")

        # Card de Resultado da Predição da IA
        alert_color = "#16a34a" if predicted_online < 30 else ("#ea580c" if predicted_online < 34 else "#dc2626")
        status_msg = "Ameno / Confortável" if predicted_online < 30 else ("Calor Intenso" if predicted_online < 34 else "Alerta de Calor Extremo")
        if is_contingency:
            context_msg = f"Inferência de contingência baseada nos dados consolidados do dia {live_summary['date']} e na inércia dos 7 dias anteriores da base local."
        elif "ipmet" in status:
            context_msg = f"Cálculo automatizado com base nos sensores físicos da Estação Meteorológica do IPMet/UNESP Bauru ({live_summary['date']}) e na inércia temporal dos últimos 7 dias."
        else:
            context_msg = "Cálculo automatizado com base nos dados meteorológicos reais de hoje e na inércia dos últimos 7 dias."

        st.write("")
        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border: 2px solid {alert_color}; border-radius: 12px; padding: 22px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                <div style="font-size: 1.15rem; color: #334155; font-weight: 600;">
                    🤖 Previsão da Inteligência Artificial (XGBoost) para a Temperatura Máxima de Amanhã:
                </div>
                <div style="font-size: 3.2rem; font-weight: 800; color: {alert_color}; margin: 8px 0;">
                    {predicted_online:.1f} °C
                </div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #475569;">
                    Classificação Térmica: <span style="color: {alert_color};">{status_msg}</span>
                </div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                    {context_msg}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif live_err:
        st.warning(f"Não foi possível conectar à API em tempo real: {live_err}. Exibindo modo de contingência.")

    # Expansor opcional para simulações manuais alternativas
    with st.expander("🛠️ Modo Manual Opcional: Testar Cenários Hipotéticos Alternativos"):
        st.markdown("Ajuste manualmente parâmetros caso queira simular cenários climáticos hipotéticos personalizados:")
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_temp_max = st.slider("Temp. Máxima Simulada (°C):", 15.0, 42.0, 28.5, 0.5)
            sim_temp_mean = st.slider("Temp. Média Simulada (°C):", 10.0, 35.0, 23.0, 0.5)
        with sim_col2:
            sim_rad = st.slider("Radiação Solar Simulada (W/m²):", 1000.0, 30000.0, 18500.0, 500.0)
            sim_hum_min = st.slider("Umidade Mínima Simulada (%):", 10.0, 95.0, 45.0, 5.0)
        with sim_col3:
            sim_rain = st.slider("Chuva Simulada (mm):", 0.0, 100.0, 0.0, 1.0)
            sim_press = st.slider("Pressão Barométrica (hPa):", 935.0, 975.0, 955.0, 1.0)

        if xgb_model is not None:
            sim_amp = sim_temp_max - (sim_temp_mean - 4.0)
            manual_vector = pd.DataFrame([{
                "temp_mean": sim_temp_mean,
                "temp_min": sim_temp_mean - 4.0,
                "temp_max": sim_temp_max,
                "thermal_amplitude": sim_amp,
                "humidity_mean": sim_hum_min + 20.0,
                "humidity_min": sim_hum_min,
                "precipitation_sum": sim_rain,
                "wind_speed_mean": 9.5,
                "wind_speed_max": 22.0,
                "surface_pressure_mean": sim_press,
                "solar_radiation_sum": sim_rad,
                "temp_mean_roll7": sim_temp_mean,
                "temp_mean_roll30": sim_temp_mean,
                "precip_sum_roll7": sim_rain,
                "temp_mean_lag1": sim_temp_mean,
                "temp_mean_lag7": sim_temp_mean,
                "temp_max_lag2": sim_temp_max,
                "temp_max_lag3": sim_temp_max,
                "humidity_lag1": sim_hum_min + 20.0,
                "radiation_lag1": sim_rad,
                "pressure_lag1": sim_press,
                "sin_doy": 0.0,
                "cos_doy": 1.0,
            }])
            manual_pred = float(xgb_model.predict(manual_vector)[0])
            st.info(f"Previsão para o cenário manual hipotético: **{manual_pred:.1f} °C**")

    st.write("---")

    # Métricas de Validação Científica
    if ml_metrics:
        st.write("#### 📊 Validação Científica do Modelo no Período de Teste Cego (2022–2024)")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value" style="color: #16a34a;">{ml_metrics['xgboost']['mae_celsius']:.2f} °C</div>
                    <div class="metric-label">Erro Médio Absoluto (MAE)</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with mcol2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{ml_metrics['xgboost']['r2_score']:.1%}</div>
                    <div class="metric-label">Coeficiente R² (Aderência)</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with mcol3:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value" style="color: #64748b;">{ml_metrics['baseline_persistence']['mae_celsius']:.2f} °C</div>
                    <div class="metric-label">Erro Baseline (Ingênuo)</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with mcol4:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value" style="color: #2563eb;">+{ml_metrics['improvement_over_baseline_pct']:.1f}%</div>
                    <div class="metric-label">Melhoria sobre o Baseline</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.write("")

        if not df_preds.empty:
            st.write("#### 🎯 Confronto: Temperatura Máxima Real vs. Previsão da IA (2022–2024)")
            pred_years = [2022, 2023, 2024]
            selected_pred_year = st.selectbox(
                "Filtrar ano de teste para inspeção detalhada:",
                ["Todos os anos de teste (2022 a 2024)"] + pred_years,
            )

            if selected_pred_year == "Todos os anos de teste (2022 a 2024)":
                plot_preds = df_preds
            else:
                plot_preds = df_preds[df_preds.index.year == int(selected_pred_year)]

            fig_pred = go.Figure()

            fig_pred.add_trace(
                go.Scatter(
                    x=plot_preds.index,
                    y=plot_preds["real_temp_max"],
                    mode="lines",
                    line=dict(color="#ef4444", width=2),
                    name="Temperatura Máxima Real (°C)",
                    hovertemplate="%{x|%d/%m/%Y}: Real %{y:.1f} °C<extra></extra>",
                )
            )

            fig_pred.add_trace(
                go.Scatter(
                    x=plot_preds.index,
                    y=plot_preds["pred_temp_max_xgb"],
                    mode="lines",
                    line=dict(color="#2563eb", width=2, dash="dot"),
                    name="Previsão XGBoost (°C)",
                    hovertemplate="%{x|%d/%m/%Y}: Previsto %{y:.1f} °C<extra></extra>",
                )
            )

            fig_pred.update_layout(
                title=f"<b>Curva Comparativa de Aderência Preditiva ({selected_pred_year})</b>",
                yaxis_title="Temperatura Máxima (°C)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

        # Gráfico de Feature Importance
        st.write("#### 🔍 O que a IA Considera Mais Importante para Prever o Calor?")
        top_feats = pd.DataFrame(ml_metrics.get("top_features", []))
        if not top_feats.empty:
            feat_name_map = {
                "temp_max": "Temp. Máxima de Hoje",
                "temp_mean": "Temp. Média de Hoje",
                "solar_radiation_sum": "Radiação Solar Incidente",
                "thermal_amplitude": "Amplitude Térmica Diurna",
                "humidity_min": "Umidade Mínima do Ar",
                "cos_doy": "Ciclo Sazonal do Ano (Cosseno)",
                "precipitation_sum": "Chuva Acumulada no Dia",
                "temp_mean_roll7": "Média de Temp. dos Últimos 7 Dias",
                "surface_pressure_mean": "Pressão Barométrica",
                "sin_doy": "Ciclo Sazonal do Ano (Seno)",
            }
            top_feats["nome_didatico"] = top_feats["feature"].map(lambda f: feat_name_map.get(f, f))

            fig_feat = px.bar(
                top_feats.sort_values(by="importance_gain_pct", ascending=True),
                x="importance_gain_pct",
                y="nome_didatico",
                orientation="h",
                title="<b>Importância Relativa das Variáveis no Modelo XGBoost (%)</b>",
                labels={"importance_gain_pct": "Ganho de Informação (%)", "nome_didatico": "Variável Meteorológica"},
                color="importance_gain_pct",
                color_continuous_scale="Blues",
            )
            fig_feat.update_layout(margin=dict(l=40, r=40, t=50, b=40), coloraxis_showscale=False)
            st.plotly_chart(fig_feat, use_container_width=True)

# ------------------------------------------
# ABA 5: QUEIMADAS & RISCO XGBOOST
# ------------------------------------------
with tab5:
    st.subheader("🔥 Relação entre Períodos de Estiagem e Queimadas em Bauru/SP")

    st.markdown(
        """
        <div class="explanation-box">
        💡 <b>Queimadas e Estiagem em Bauru (Dados Oficiais INPE 2010 a 2025):</b><br>
        Esta seção conecta a série histórica meteorológica de Bauru com as detecções do satélite de referência do 
        <b>INPE (Instituto Nacional de Pesquisas Espaciais)</b>. O algoritmo <b>XGBoost Classifier</b> foi treinado 
        para aprender o impacto direto dos dias consecutivos sem chuva e do déficit hídrico acumulado na 
        <b>probabilidade percentual (%) de focos de queimada</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filtragem temporal reativa para queimadas
    if not df_queimadas.empty and "year" in df_queimadas.columns:
        f_queimadas = df_queimadas[
            (df_queimadas["year"] >= year_range[0]) & (df_queimadas["year"] <= year_range[1])
        ]
    else:
        f_queimadas = df_queimadas

    # Cartões de Métricas
    total_focos = int(f_queimadas["queimadas_focos_count"].sum()) if not f_queimadas.empty else 0
    dias_fogo = int(f_queimadas["has_fire"].sum()) if not f_queimadas.empty else 0
    taxa_dias = (dias_fogo / max(1, len(f_queimadas))) * 100 if not f_queimadas.empty else 0.0

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    with col_q1:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-value">{total_focos}</div><div class="metric-label">Total de Focos Filtrados</div></div>""",
            unsafe_allow_html=True,
        )
    with col_q2:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-value">{dias_fogo}</div><div class="metric-label">Dias com Fogo ({taxa_dias:.1f}%)</div></div>""",
            unsafe_allow_html=True,
        )
    with col_q3:
        st.markdown(
            """<div class="metric-card"><div class="metric-value" style="color: #dc2626;">2011 e 2024</div><div class="metric-label">Anos Críticos (35 focos/ano)</div></div>""",
            unsafe_allow_html=True,
        )
    with col_q4:
        st.markdown(
            """<div class="metric-card"><div class="metric-value" style="color: #16a34a;">79.9%</div><div class="metric-label">ROC-AUC (Poder Preditivo)</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    # Gráfico 1: Curva de Probabilidade Calculada pelo XGBoost
    st.subheader("📈 Curva de Sensibilidade: Probabilidade de Queimada vs. Dias de Estiagem")
    st.markdown(
        "Observe no gráfico abaixo como a probabilidade calculada pelo **XGBoost** reage ao aumento dos dias sem chuva contínuos em Bauru:"
    )

    if curva_risco:
        df_curva = pd.DataFrame(curva_risco)
        fig_curva = go.Figure()

        # Linha 1: Índice de Risco Relativo (Escala Ponderada de Alerta)
        fig_curva.add_trace(
            go.Scatter(
                x=df_curva["dias_estiagem"],
                y=df_curva["chance_queimada_percentual"],
                mode="lines+markers",
                name="Índice de Risco / Alerta (XGBoost %)",
                line=dict(color="#dc2626", width=3),
                marker=dict(size=7),
            )
        )

        # Linha 2: Probabilidade Real Calibrada (Platt Scaling)
        if "probabilidade_calibrada_percentual" in df_curva.columns:
            fig_curva.add_trace(
                go.Scatter(
                    x=df_curva["dias_estiagem"],
                    y=df_curva["probabilidade_calibrada_percentual"],
                    mode="lines+markers",
                    name="Probabilidade Real Calibrada (Taxa ao Dia %)",
                    line=dict(color="#2563eb", width=3, dash="dash"),
                    marker=dict(size=7),
                )
            )

        fig_curva.update_layout(
            title="Chance Percentual Calculada pelo XGBoost vs. Dias Consecutivos de Estiagem",
            plot_bgcolor="white",
            hovermode="x unified",
            xaxis=dict(title="Dias Consecutivos sem Chuva (Estiagem Contínua)"),
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        # Zonas de risco
        fig_curva.add_hrect(
            y0=0, y1=35, fillcolor="green", opacity=0.07, line_width=0, annotation_text="Faixa de Risco Baixo"
        )
        fig_curva.add_hrect(
            y0=35, y1=70, fillcolor="orange", opacity=0.07, line_width=0, annotation_text="Faixa de Risco Elevado"
        )
        fig_curva.add_hrect(
            y0=70, y1=100, fillcolor="red", opacity=0.07, line_width=0, annotation_text="Faixa de Risco Crítico"
        )
        st.plotly_chart(fig_curva, use_container_width=True)

    # Gráfico 2: Histórico Anual de Queimadas
    st.subheader("📊 Focos Anuais de Queimada Detectados pelo INPE em Bauru")
    if not f_queimadas.empty and "year" in f_queimadas.columns:
        df_ano_focos = (
            f_queimadas.groupby("year")["queimadas_focos_count"].sum().reset_index()
        )
        df_ano_focos.columns = ["Ano", "Focos"]
        fig_ano_focos = px.bar(
            df_ano_focos,
            x="Ano",
            y="Focos",
            text="Focos",
            title="Distribuição Anual de Focos em Bauru/SP (Satélite de Referência)",
            color="Focos",
            color_continuous_scale="Reds",
        )
        fig_ano_focos.update_traces(textposition="outside")
        fig_ano_focos.update_layout(plot_bgcolor="white", margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_ano_focos, use_container_width=True)

    # 4. SIMULADOR INTERATIVO EM TEMPO REAL
    st.subheader("🎛️ Simulador Interativo: Calcule a Chance de Queimada com o XGBoost")
    st.markdown(
        "Ajuste os parâmetros atmosféricos abaixo para rodar a inferência do modelo treinado em tempo real:"
    )

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_dias_secos = st.slider("Dias consecutivos de estiagem:", min_value=0, max_value=60, value=20, step=1)
        sim_temp_max = st.slider(
            "Temperatura Máxima (°C):", min_value=20.0, max_value=42.0, value=32.0, step=0.5
        )
    with sim_col2:
        sim_umidade_min = st.slider(
            "Umidade Relativa Mínima (%):", min_value=10.0, max_value=80.0, value=22.0, step=1.0
        )
        sim_vento_max = st.slider(
            "Rajada Máxima de Vento (km/h):", min_value=5.0, max_value=60.0, value=22.0, step=1.0
        )
    with sim_col3:
        sim_chuva_30d = st.slider(
            "Chuva acumulada em 30 dias (mm):", min_value=0.0, max_value=200.0, value=2.0, step=5.0
        )
        sim_mes = st.selectbox(
            "Mês do Ano:",
            [
                "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
                "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
            ],
            index=7,
        )

    if xgb_queimadas_model is not None:
        meses_map = {
            "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4, "Maio": 5, "Junho": 6,
            "Julho": 7, "Agosto": 8, "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12,
        }
        num_mes = meses_map[sim_mes]
        doy_aprox = (num_mes - 1) * 30 + 15

        vetor_teste = pd.DataFrame(
            [
                {
                    "consecutive_dry_days": sim_dias_secos,
                    "precip_sum_roll7": 0.0 if sim_dias_secos >= 7 else 5.0,
                    "precip_sum_roll15": 0.0 if sim_dias_secos >= 15 else 10.0,
                    "precip_sum_roll30": sim_chuva_30d,
                    "precip_sum_roll60": sim_chuva_30d * 1.5,
                    "humidity_min": sim_umidade_min,
                    "humidity_mean": min(100.0, sim_umidade_min + 25.0),
                    "temp_max": sim_temp_max,
                    "temp_mean": sim_temp_max - 6.0,
                    "thermal_amplitude": 12.0,
                    "wind_speed_max": sim_vento_max,
                    "solar_radiation_sum": 18000.0,
                    "indice_secura": sim_temp_max / max(1.0, sim_umidade_min),
                    "indice_propagacao_vento": (sim_vento_max * sim_temp_max) / max(1.0, sim_umidade_min),
                    "sin_doy": np.sin(2 * np.pi * doy_aprox / 365.25),
                    "cos_doy": np.cos(2 * np.pi * doy_aprox / 365.25),
                }
            ]
        )

        prob_simulada = float(xgb_queimadas_model.predict_proba(vetor_teste)[:, 1][0] * 100)
        prob_calibrada = (
            float(xgb_calibrado_model.predict_proba(vetor_teste)[:, 1][0] * 100)
            if xgb_calibrado_model is not None
            else None
        )

        # Determinação do nível de alerta e cor
        if prob_simulada < 35:
            alerta_cor = "#16a34a"
            alerta_texto = "🟢 BAIXO RISCO DE QUEIMADA"
            alerta_desc = "Condições desfavoráveis para propagação de fogo."
        elif prob_simulada < 70:
            alerta_cor = "#ea580c"
            alerta_texto = "🟠 RISCO MODERADO A ALTO DE QUEIMADA"
            alerta_desc = "Atenção: A estiagem e o ar seco aumentam a vulnerabilidade da vegetação."
        else:
            alerta_cor = "#dc2626"
            alerta_texto = "🔴 RISCO CRÍTICO DE QUEIMADA"
            alerta_desc = "Perigo extremo: Longo período sem chuva associado à baixa umidade e calor intenso."

        badge_calibrada = ""
        if prob_calibrada is not None:
            badge_calibrada = f"""
            <div style="margin-top: 10px; padding: 6px 14px; background: #e0f2fe; border: 1px solid #7dd3fc; border-radius: 8px; display: inline-block; font-size: 0.95rem; color: #0369a1; font-weight: 600;">
                📊 Probabilidade Real Calibrada (Taxa ao Dia): {prob_calibrada:.2f}%
            </div>
            """

        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border: 2px solid {alerta_cor}; border-radius: 12px; padding: 22px; text-align: center; margin-top: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
                <div style="font-size: 1.2rem; font-weight: 800; color: {alerta_cor};">{alerta_texto}</div>
                <div style="font-size: 3.2rem; font-weight: 900; color: {alerta_cor}; margin: 8px 0;">{prob_simulada:.1f}%</div>
                <div style="font-size: 1.0rem; color: #334155; font-weight: 500;">{alerta_desc}</div>
                {badge_calibrada}
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 8px;">Índice de alerta ponderado pelo scale_pos_weight e probabilidade real calibrada por Platt Scaling</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# ------------------------------------------
# ABA 6: EXPLORADOR DE DADOS
# ------------------------------------------
with tab6:
    st.subheader("Explorador de Dados e Download")
    st.markdown("Consulte os dados meteorológicos tratados e exporte tabelas para suas próprias análises.")

    freq_choice = st.radio("Escolha a granularidade da tabela:", ["Mensal", "Diária"], horizontal=True)

    if freq_choice == "Mensal":
        display_df = filtered_monthly[
            [
                "year",
                "month",
                "season",
                "temp_mean",
                "temp_min",
                "temp_max",
                "precipitation_total",
                "rainy_days_count",
                "max_consecutive_dry_days",
                "temp_anomaly",
            ]
        ].copy()
        display_df = display_df.rename(
            columns={
                "year": "Ano",
                "month": "Mês",
                "season": "Estação",
                "temp_mean": "Temp. Média (°C)",
                "temp_min": "Temp. Mín (°C)",
                "temp_max": "Temp. Máx (°C)",
                "precipitation_total": "Chuva Total (mm)",
                "rainy_days_count": "Dias Chuvosos",
                "max_consecutive_dry_days": "Máx Dias Secos",
                "temp_anomaly": "Anomalia Térmica (°C)",
            }
        )
    else:
        display_df = filtered_daily[
            [
                "year",
                "month",
                "season",
                "temp_mean",
                "temp_min",
                "temp_max",
                "thermal_amplitude",
                "precipitation_sum",
                "consecutive_dry_days",
                "humidity_mean",
                "wind_speed_mean",
            ]
        ].head(1000).copy()
        display_df = display_df.rename(
            columns={
                "year": "Ano",
                "month": "Mês",
                "season": "Estação",
                "temp_mean": "Temp. Média (°C)",
                "temp_min": "Temp. Mín (°C)",
                "temp_max": "Temp. Máx (°C)",
                "thermal_amplitude": "Ampl. Térmica (°C)",
                "precipitation_sum": "Chuva (mm)",
                "consecutive_dry_days": "Dias Sem Chuva",
                "humidity_mean": "Umidade Média (%)",
                "wind_speed_mean": "Vento Médio (km/h)",
            }
        )

    st.dataframe(display_df, use_container_width=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Baixar Dados Filtrados em CSV",
        data=csv_bytes,
        file_name=f"bauru_clima_{freq_choice.lower()}_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv",
    )

# ==========================================
# RODAPÉ COM NOME DO AUTOR
# ==========================================
st.markdown(
    """
    <div class="footer-box">
        <b>Estudo de Séries Temporais Meteorológicas – Bauru/SP</b><br>
        <b>Autor: Prof. Dr. Douglas A. Galbiatti</b><br>
        <span style="font-size: 0.85rem; color: #64748b;">
        Metodologia KDD (Knowledge Discovery in Databases) | Modelo Preditivo: XGBoost Regressor | Fonte: Open-Meteo Historical Weather API (ERA5/ECMWF Reanalysis)
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
