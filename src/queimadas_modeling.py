"""
Módulo de Machine Learning (KDD - Etapa 5)
Treina modelo XGBoost Classifier para estimar a probabilidade percentual de queimada
em Bauru em função do aumento dos períodos de estiagem e condições meteorológicas.
Gera a curva de sensibilidade (Probabilidade vs. Dias Consecutivos de Estiagem).
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb


# Definição dos diretórios de modelos e relatórios
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/tables")
PROCESSED_FILE = Path("data/processed/bauru_weather_queimadas_daily.parquet")


def preparar_matriz_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Seleciona as variáveis meteorológicas e de estiagem para treinar o modelo.
    Variável Alvo (y): has_fire (1 = dia com queimada, 0 = sem queimada).
    """
    df_data = df.copy()

    # Ciclo senoidal sazonal (dia do ano) para capturar a época de seca (inverno/primavera)
    doy = df_data.index.dayofyear
    df_data["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df_data["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    # Features selecionadas
    features = [
        "consecutive_dry_days",  # Dias consecutivos de estiagem (núcleo da análise)
        "precip_sum_roll7",  # Chuva acumulada 7 dias
        "precip_sum_roll15",  # Chuva acumulada 15 dias
        "precip_sum_roll30",  # Chuva acumulada 30 dias
        "precip_sum_roll60",  # Chuva acumulada 60 dias (seca prolongada)
        "humidity_min",  # Umidade relativa mínima do ar (%)
        "humidity_mean",  # Umidade relativa média (%)
        "temp_max",  # Temperatura máxima do dia (°C)
        "temp_mean",  # Temperatura média (°C)
        "thermal_amplitude",  # Amplitude térmica (°C)
        "wind_speed_max",  # Rajada máxima de vento (km/h)
        "solar_radiation_sum",  # Radiação solar total
        "indice_secura",  # temp_max / humidity_min
        "indice_propagacao_vento",  # (wind * temp) / humidity
        "sin_doy",  # Sazonalidade
        "cos_doy",
    ]

    X = df_data[features].copy()
    y = df_data["has_fire"].copy()

    return X, y


def treinar_xgboost_queimadas() -> Dict[str, Any]:
    """
    Treina o XGBoost com divisão cronológica estrita (Treino: <= 2021 | Teste: >= 2022)
    e gera a curva de probabilidade percentual em relação aos dias de estiagem.
    """
    print(f"Carregando dados consolidados de: {PROCESSED_FILE}")
    df = pd.read_parquet(PROCESSED_FILE)

    X, y = preparar_matriz_features(df)

    # Divisão Cronológica de Séries Temporais (sem vazamento de dados futuros)
    treino_idx = X.index.year <= 2021
    teste_idx = X.index.year >= 2022

    X_train, y_train = X[treino_idx], y[treino_idx]
    X_test, y_test = X[teste_idx], y[teste_idx]

    print(f"Amostras de Treino: {len(X_train)} dias (2010 a 2021)")
    print(f"Amostras de Teste:  {len(X_test)} dias (2022 a 2024+)")
    print(f"Taxa de queimadas no treino: {y_train.mean()*100:.2f}% | no teste: {y_test.mean()*100:.2f}%")

    # Tratamento de desbalanceamento de classes:
    # Como queimadas ocorrem em ~2% dos dias, calculamos a razão negativa/positiva
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = n_neg / max(n_pos, 1)

    # Configuração do XGBoost para Classificação Probabilística
    modelo = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        scale_pos_weight=scale_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    print("Treinando modelo XGBoost...")
    modelo.fit(X_train, y_train)

    # Previsão de probabilidades no conjunto de teste (classe 1 = queimada)
    prob_teste = modelo.predict_proba(X_test)[:, 1]

    # Cálculo das métricas de performance
    auc_roc = roc_auc_score(y_test, prob_teste)
    pr_auc = average_precision_score(y_test, prob_teste)
    brier = brier_score_loss(y_test, prob_teste)

    print(f"\n[MÉTRICAS DO MODELO - TESTE]")
    print(f" -> ROC-AUC: {auc_roc:.4f}")
    print(f" -> PR-AUC (Precisão Média): {pr_auc:.4f}")
    print(f" -> Brier Score: {brier:.4f}")

    # -------------------------------------------------------------
    # Cálculo da Curva Percentual: Estiagem vs. Chance de Queimada
    # -------------------------------------------------------------
    print("\nCalculando a curva percentual de chance de queimada vs. dias de estiagem...")

    # Criamos um cenário meteorológico típico de período seco em Bauru:
    # Inverno/Primavera: temp_max = 31°C, umidade_min = 25%, vento = 20 km/h, sem chuva recente
    faixa_dias_secos = np.arange(0, 61, 2)  # De 0 a 60 dias sem chuva (de 2 em 2 dias)
    curva_risco = []

    # Valores medianos do período seco para as outras variáveis
    for dias in faixa_dias_secos:
        cenario = pd.DataFrame(
            [
                {
                    "consecutive_dry_days": dias,
                    "precip_sum_roll7": 0.0 if dias >= 7 else 10.0,
                    "precip_sum_roll15": 0.0 if dias >= 15 else 15.0,
                    "precip_sum_roll30": 0.0 if dias >= 30 else 25.0,
                    "precip_sum_roll60": 0.0 if dias >= 60 else 50.0,
                    "humidity_min": max(15.0, 45.0 - (dias * 0.5)),  # Umidade cai com a seca
                    "humidity_mean": max(35.0, 65.0 - (dias * 0.5)),
                    "temp_max": 31.0,
                    "temp_mean": 24.0,
                    "thermal_amplitude": 14.0,
                    "wind_speed_max": 20.0,
                    "solar_radiation_sum": 18000.0,
                    "indice_secura": 31.0 / max(15.0, 45.0 - (dias * 0.5)),
                    "indice_propagacao_vento": (20.0 * 31.0) / max(15.0, 45.0 - (dias * 0.5)),
                    "sin_doy": -0.85,  # Agosto/Setembro (pico da seca em SP)
                    "cos_doy": -0.52,
                }
            ]
        )

        # O predict_proba retorna [P(não), P(sim)]. Multiplicamos por 100 para ter a %
        chance_percentual = float(modelo.predict_proba(cenario)[:, 1][0] * 100)
        curva_risco.append({"dias_estiagem": int(dias), "chance_queimada_percentual": round(chance_percentual, 2)})

    # Salva o modelo treinado e os artefatos
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    modelo_path = MODELS_DIR / "xgboost_queimadas_bauru.joblib"
    curva_path = REPORTS_DIR / "curva_estiagem_vs_queimadas.json"
    metricas_path = REPORTS_DIR / "queimadas_metricas.json"

    joblib.dump(modelo, modelo_path)

    with open(curva_path, "w", encoding="utf-8") as f:
        json.dump(curva_risco, f, indent=4, ensure_ascii=False)

    metricas = {
        "roc_auc": round(auc_roc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "total_treino": int(len(X_train)),
        "total_teste": int(len(X_test)),
    }
    with open(metricas_path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4)

    print(f"\nModelo salvo com sucesso em: {modelo_path}")
    print(f"Curva percentual salva em: {curva_path}")
    return metricas


if __name__ == "__main__":
    treinar_xgboost_queimadas()
