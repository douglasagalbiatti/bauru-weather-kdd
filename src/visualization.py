"""
Módulo de Visualização (Etapas 5 e 6 - KDD)
Responsável pela geração de gráficos analíticos e climáticos em alta resolução
para o diretório reports/figures/.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import STL

# Estilo global refinado
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_climograph(df_monthly: pd.DataFrame, output_path: Path) -> None:
    """
    Plota o Climograma Climatológico de Bauru/SP (2010-2024).
    """
    climatology = df_monthly.groupby("month").agg(
        precip_mean=("precipitation_total", "mean"),
        temp_mean=("temp_mean", "mean"),
    )
    month_labels = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)
    ax2 = ax1.twinx()

    # Barras de precipitação
    bars = ax1.bar(
        climatology.index,
        climatology["precip_mean"],
        color="#2b5c8f",
        alpha=0.75,
        width=0.55,
        label="Precipitação Média (mm)",
    )

    # Linha de temperatura
    line = ax2.plot(
        climatology.index,
        climatology["temp_mean"],
        color="#d95f02",
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="Temperatura Média (°C)",
    )

    ax1.set_xlabel("Mês do Ano", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Precipitação Acumulada Média (mm)", fontsize=11, color="#2b5c8f", fontweight="bold")
    ax2.set_ylabel("Temperatura Média (°C)", fontsize=11, color="#d95f02", fontweight="bold")
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_labels)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    plt.title("Climograma Climatológico – Bauru/SP (2010–2024)", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_temperature_anomalies(df_monthly: pd.DataFrame, output_path: Path) -> None:
    """
    Plota as anomalias mensais de temperatura (diferença em relação à climatologia).
    """
    fig, ax = plt.subplots(figsize=(13, 5), dpi=200)

    dates = df_monthly.index
    anomalies = df_monthly["temp_anomaly"]
    colors = np.where(anomalies >= 0, "#d73027", "#4575b4")

    ax.bar(dates, anomalies, width=25, color=colors, alpha=0.85, label="Anomalia Mensal")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)

    # Média móvel de 12 meses da anomalia
    roll12 = anomalies.rolling(window=12, center=True).mean()
    ax.plot(dates, roll12, color="#1a1a1a", linewidth=1.8, label="Tendência (Média Móvel 12m)")

    ax.set_title("Anomalias Mensais de Temperatura do Ar – Bauru/SP (2010–2024)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Anomalia de Temperatura (°C)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_monthly_boxplots(df_daily: pd.DataFrame, output_path: Path) -> None:
    """
    Plota boxplots da temperatura média diária agrupada por mês.
    """
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)

    month_labels = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    df_plot = df_daily.copy()
    df_plot["month_name"] = df_plot["month"].map(lambda m: month_labels[m - 1])

    sns.boxplot(
        data=df_plot,
        x="month_name",
        y="temp_mean",
        palette="coolwarm",
        ax=ax,
        fliersize=3,
        linewidth=1.2,
    )

    ax.set_title("Distribuição Sazonal da Temperatura Média Diária – Bauru/SP", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Mês", fontsize=11, fontweight="bold")
    ax.set_ylabel("Temperatura Média Diária (°C)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_correlation_heatmap(df_daily: pd.DataFrame, output_path: Path) -> None:
    """
    Plota matriz de correlação das variáveis meteorológicas diárias.
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)

    vars_subset = [
        "temp_mean",
        "thermal_amplitude",
        "humidity_mean",
        "precipitation_sum",
        "wind_speed_mean",
        "surface_pressure_mean",
        "solar_radiation_sum",
    ]
    labels = [
        "Temp. Média",
        "Ampl. Térmica",
        "Umidade Média",
        "Precipitação",
        "Vento Médio",
        "Pressão Superf.",
        "Radiação Solar",
    ]

    corr = df_daily[vars_subset].corr()
    corr.columns = labels
    corr.index = labels

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Matriz de Correlação Meteorológica (Escala Diária)", fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_stl_decomposition(df_monthly: pd.DataFrame, output_path: Path) -> None:
    """
    Plota os 4 componentes da decomposição STL da temperatura média mensal.
    """
    series = df_monthly["temp_mean"].dropna()
    stl = STL(series, period=12, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), dpi=200, sharex=True)

    axes[0].plot(series.index, series.values, color="#2b5c8f", lw=1.2)
    axes[0].set_ylabel("Observado (°C)", fontweight="bold")
    axes[0].set_title("Decomposição STL da Temperatura Média Mensal – Bauru/SP", fontsize=13, fontweight="bold", pad=10)

    axes[1].plot(series.index, res.trend, color="#d95f02", lw=2.0)
    axes[1].set_ylabel("Tendência (°C)", fontweight="bold")

    axes[2].plot(series.index, res.seasonal, color="#7570b3", lw=1.2)
    axes[2].set_ylabel("Sazonalidade (°C)", fontweight="bold")

    axes[3].scatter(series.index, res.resid, color="#666666", s=10, alpha=0.7)
    axes[3].axhline(0, color="black", linestyle="--", lw=0.8)
    axes[3].set_ylabel("Resíduo (°C)", fontweight="bold")
    axes[3].set_xlabel("Ano", fontweight="bold")

    for ax in axes:
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_drought_events(df_daily: pd.DataFrame, output_path: Path) -> None:
    """
    Plota os episódios de estiagem prolongada (dias consecutivos sem chuva).
    """
    fig, ax = plt.subplots(figsize=(13, 5), dpi=200)

    ax.plot(df_daily.index, df_daily["consecutive_dry_days"], color="#d95f02", lw=1.2, label="Dias Consecutivos Secos")
    ax.axhline(30, color="#feb24c", linestyle="--", lw=1.0, label="Alerta Estiagem (30 dias)")
    ax.axhline(60, color="#bd0026", linestyle="--", lw=1.2, label="Estiagem Crítica (60 dias)")

    ax.set_title("Evolução Histórica de Estiagens Contínuas – Bauru/SP (2010–2024)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Dias Consecutivos Sem Chuva", fontsize=11, fontweight="bold")
    ax.set_xlabel("Ano", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_visualizations(
    daily_parquet: str = "data/processed/bauru_weather_daily.parquet",
    monthly_parquet: str = "data/processed/bauru_weather_monthly.parquet",
    output_dir: str = "reports/figures",
) -> None:
    """
    Carrega os dados e gera toda a suíte de figuras analíticas do projeto.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Carregando datasets processados para plotagem...")
    df_daily = pd.read_parquet(daily_parquet)
    df_monthly = pd.read_parquet(monthly_parquet)

    # As figuras descrevem a janela fixa do estudo (2010–2024), como dizem os títulos
    df_daily = df_daily[(df_daily.index.year >= 2010) & (df_daily.index.year <= 2024)]
    df_monthly = df_monthly[(df_monthly.index.year >= 2010) & (df_monthly.index.year <= 2024)]

    print("1. Gerando Climograma...")
    plot_climograph(df_monthly, out_path / "fig01_climograma_bauru.png")

    print("2. Gerando Gráfico de Anomalias de Temperatura...")
    plot_temperature_anomalies(df_monthly, out_path / "fig02_anomalias_temperatura.png")

    print("3. Gerando Boxplots Sazonais...")
    plot_monthly_boxplots(df_daily, out_path / "fig03_boxplots_temperatura_mensal.png")

    print("4. Gerando Matriz de Correlação...")
    plot_correlation_heatmap(df_daily, out_path / "fig04_heatmap_correlacoes.png")

    print("5. Gerando Decomposição STL...")
    plot_stl_decomposition(df_monthly, out_path / "fig05_decomposicao_stl_temperatura.png")

    print("6. Gerando Gráfico de Eventos de Estiagem...")
    plot_drought_events(df_daily, out_path / "fig06_eventos_estiagem_dias_secos.png")

    print(f"Todas as 6 figuras geradas com sucesso em: {out_path}")


if __name__ == "__main__":
    generate_all_visualizations()
