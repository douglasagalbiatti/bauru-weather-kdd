"""
Comparação entre a estação de superfície IPMet/UNESP Bauru (sensor)
e a API Open-Meteo Forecast (modelo), em médias horárias.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

FUSO = "America/Sao_Paulo"
LATITUDE, LONGITUDE = -22.3147, -49.0608

# Nome no Open-Meteo -> nome no IPMet
COLUNAS = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "surface_pressure": "pressure",
    "wind_speed_10m": "wind_speed",
}

TITULOS = {
    "temp": "Temperatura (°C)",
    "humidity": "Umidade relativa (%)",
    "pressure": "Pressão na superfície (hPa)",
    "wind_speed": "Vento (km/h)",
}


def carregar_ipmet(caminho_csv: Path) -> pd.DataFrame:
    """Lê o histórico do coletor, com índice de data/hora no fuso de Brasília."""
    df = pd.read_csv(caminho_csv)
    df["reading_time"] = pd.to_datetime(df["reading_time"], utc=True).dt.tz_convert(FUSO)
    return df.set_index("reading_time").sort_index()


def buscar_open_meteo(dias: int = 7) -> tuple[pd.DataFrame, float]:
    """Busca a série horária do Open-Meteo dos últimos `dias` dias. Retorna (dados, altitude do modelo)."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(COLUNAS),
        "timezone": FUSO,
        "past_days": dias,
        "forecast_days": 1,
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(FUSO)
    return df.set_index("time").rename(columns=COLUNAS), payload.get("elevation")


def montar_comparacao(ipmet: pd.DataFrame, om: pd.DataFrame) -> pd.DataFrame:
    """Junta as duas fontes por hora e calcula a diferença IPMet − Open-Meteo."""
    variaveis = list(COLUNAS.values())

    agora = pd.Timestamp.now(tz=FUSO)
    om = om[om.index <= agora]

    ipmet_h = ipmet[variaveis].resample("h").mean().dropna(how="all")
    comp = ipmet_h.join(om[variaveis], how="inner", lsuffix="_ipmet", rsuffix="_om")

    for var in variaveis:
        comp[f"dif_{var}"] = comp[f"{var}_ipmet"] - comp[f"{var}_om"]
    return comp


def grafico_comparacao(comp: pd.DataFrame) -> go.Figure:
    """Quatro painéis empilhados: sensor (azul, contínuo) × modelo (laranja, tracejado)."""
    fig = make_subplots(
        rows=len(TITULOS), cols=1, shared_xaxes=True,
        subplot_titles=list(TITULOS.values()), vertical_spacing=0.06,
    )
    for i, var in enumerate(TITULOS, start=1):
        fig.add_trace(
            go.Scatter(x=comp.index, y=comp[f"{var}_ipmet"], name="IPMet (sensor)",
                       mode="lines+markers", line=dict(color="#2563eb"),
                       legendgroup="ipmet", showlegend=(i == 1)),
            row=i, col=1,
        )
        fig.add_trace(
            go.Scatter(x=comp.index, y=comp[f"{var}_om"], name="Open-Meteo (modelo)",
                       mode="lines+markers", line=dict(color="#f97316", dash="dash"),
                       legendgroup="om", showlegend=(i == 1)),
            row=i, col=1,
        )
    fig.update_layout(height=900, hovermode="x unified",
                      legend=dict(orientation="h", y=1.04, x=0))
    return fig