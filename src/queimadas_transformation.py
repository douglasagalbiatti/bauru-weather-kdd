"""
Módulo de Transformação e Fusão (KDD - Etapa 4)
Une os dados meteorológicos diários de Bauru aos focos de queimada do INPE,
cria métricas de estiagem (dias secos, déficit hídrico) e prepara a base para o XGBoost.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# Caminhos dos arquivos de entrada e saída
PROCESSED_DIR = Path("data/processed")
INTERIM_DIR = Path("data/interim")


def fundir_clima_e_queimadas(
    caminho_clima: Path = PROCESSED_DIR / "bauru_weather_daily.parquet",
    caminho_queimadas: Path = INTERIM_DIR / "bauru_queimadas_diario.parquet",
) -> pd.DataFrame:
    """
    Realiza o merge diário entre a série climática e a série de queimadas de Bauru.
    Dias sem registro de foco no INPE são preenchidos com 0 (ausência de queimada).
    """
    print(f"Lendo base meteorológica diária: {caminho_clima}")
    df_clima = pd.read_parquet(caminho_clima)

    # Garante que o índice temporal do clima seja do tipo Date simples
    if not isinstance(df_clima.index, pd.DatetimeIndex):
        df_clima.index = pd.to_datetime(df_clima.index)

    # Cria uma coluna explícita 'data' para viabilizar o merge
    df_clima["data"] = pd.to_datetime(df_clima.index.date)

    print(f"Lendo base de queimadas: {caminho_queimadas}")
    df_queimadas = pd.read_parquet(caminho_queimadas)
    df_queimadas["data"] = pd.to_datetime(df_queimadas["data"])

    # Merge à esquerda: mantemos TODOS os dias do calendário climático de Bauru
    df_merged = pd.merge(df_clima, df_queimadas, on="data", how="left")

    # Se o INPE não registrou foco no dia, a contagem foi 0
    df_merged["queimadas_focos_count"] = df_merged["queimadas_focos_count"].fillna(0).astype(int)

    # Variável binária: 1 se houve pelo menos um foco de queimada no dia, 0 caso contrário
    df_merged["has_fire"] = (df_merged["queimadas_focos_count"] > 0).astype(int)

    # Restaura o índice temporal
    df_merged.set_index("data", inplace=True)
    return df_merged


def gerar_features_estiagem_e_fogo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera as variáveis preditivas de estiagem e risco de ignição:
    - Acumulados de chuva em 15 e 60 dias (déficit hídrico)
    - Razão Temperatura / Umidade (Índice de Secura Atmosférica)
    - Alvo antecipado t+1 (chance de queimada amanhã)
    """
    dados = df.copy()

    # 1. Déficits hídricos em múltiplas janelas temporais
    # A precipitação acumulada em 15 e 60 dias complementa os 7 e 30 dias já existentes
    dados["precip_sum_roll15"] = dados["precipitation_sum"].rolling(window=15, min_periods=1).sum()
    dados["precip_sum_roll60"] = dados["precipitation_sum"].rolling(window=60, min_periods=1).sum()

    # 2. Índice Combinado de Severidade de Secura:
    # Quanto maior a temperatura e menor a umidade, maior a facilidade de ignição
    dados["indice_secura"] = dados["temp_max"] / (dados["humidity_min"].clip(lower=1.0))

    # 3. Potencial de Propagação pelo Vento:
    # Vento forte associado à baixa umidade acelera queimadas
    dados["indice_propagacao_vento"] = (dados["wind_speed_max"] * dados["temp_max"]) / (
        dados["humidity_min"].clip(lower=1.0)
    )

    # 4. Alvo Preditivo para Previsão Antecipada (t+1):
    # Permite ao XGBoost prever percentualmente a chance de queimada no DIA SEGUINTE
    dados["target_fire_next_day"] = dados["has_fire"].shift(-1)

    # Remove o último dia que fica sem alvo t+1 devido ao shift
    dados_limpos = dados.dropna(subset=["target_fire_next_day"]).copy()
    dados_limpos["target_fire_next_day"] = dados_limpos["target_fire_next_day"].astype(int)

    return dados_limpos


def executar_pipeline_transformacao() -> pd.DataFrame:
    """
    Orquestra a fusão e o salvamento da base enriquecida em formato Parquet e CSV.
    """
    df_merged = fundir_clima_e_queimadas()
    df_final = gerar_features_estiagem_e_fogo(df_merged)

    # Salvamento das bases processadas prontas para a modelagem
    saida_parquet = PROCESSED_DIR / "bauru_weather_queimadas_daily.parquet"
    saida_csv = PROCESSED_DIR / "bauru_weather_queimadas_daily.csv"

    df_final.to_parquet(saida_parquet)
    df_final.to_csv(saida_csv)

    print(f"\n[SUCESSO] Base final para modelagem salva:")
    print(f" -> {saida_parquet}")
    print(f" -> {saida_csv}")
    print(f"Total de registros diários: {len(df_final)}")
    print(f"Dias com queimadas registradas: {df_final['has_fire'].sum()} ({df_final['has_fire'].mean()*100:.2f}%)")

    return df_final


if __name__ == "__main__":
    executar_pipeline_transformacao()
