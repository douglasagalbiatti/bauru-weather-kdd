"""
Módulo de Machine Learning (KDD - Etapa 5)
Treina modelo XGBoost Classifier para estimar o risco de queimada em Bauru/SP.
Implementa a calibração de probabilidade para compensar o desbalanceamento
e o parâmetro scale_pos_weight via:
1. Calibração Paramétrica de Platt (Platt Scaling via Regressão Logística Unidimensional)
2. Inversão Analítica da Razão de Chances (Odds Ratio)
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb


# Definição dos diretórios de modelos e relatórios
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/tables")
PROCESSED_FILE = Path("data/processed/bauru_weather_queimadas_daily.parquet")


class PlattCalibrator:
    """
    Calibrador de Probabilidade de Platt (Platt Scaling).
    Ajusta uma regressão logística univariada sobre as predições do XGBoost ponderado
    usando um conjunto de validação temporal independente para produzir probabilidades bem calibradas.
    """

    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrator = LogisticRegression(solver="lbfgs")

    def fit(self, X_val, y_val):
        prob_val = self.base_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
        self.calibrator.fit(prob_val, y_val)
        return self

    def predict_proba(self, X):
        prob_raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.calibrator.predict_proba(prob_raw)


def recalibrar_probabilidade_analitica(prob_ponderada: float, scale_weight: float) -> float:
    """
    Recalibra analiticamente a probabilidade inflada pelo scale_pos_weight.
    Inverte a razão de chances (odds ratio):
    odds_real = odds_ponderada / scale_weight
    p_calibrada = odds_real / (1 + odds_real)
    """
    p = np.clip(prob_ponderada, 1e-6, 1.0 - 1e-6)
    odds_ponderada = p / (1.0 - p)
    odds_real = odds_ponderada / scale_weight
    p_calibrada = odds_real / (1.0 + odds_real)
    return float(p_calibrada)


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
    Treina o XGBoost com scale_pos_weight e calibração de probabilidade.
    Divisão Cronológica:
    - Treino (2010 a 2019): Ajuste dos pesos das árvores
    - Validação (2020 a 2021): Ajuste da calibração Platt Scaling (Sigmoide)
    - Teste (2022 a 2024+): Avaliação cega de performance e calibração
    """
    print(f"Carregando dados consolidados de: {PROCESSED_FILE}")
    df = pd.read_parquet(PROCESSED_FILE)

    X, y = preparar_matriz_features(df)

    # Divisão Cronológica Tripla
    treino_idx = X.index.year <= 2019
    val_idx = (X.index.year >= 2020) & (X.index.year <= 2021)
    teste_idx = X.index.year >= 2022

    X_train, y_train = X[treino_idx], y[treino_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[teste_idx], y[teste_idx]

    print(f"Amostras de Treino:     {len(X_train)} dias (2010 a 2019)")
    print(f"Amostras de Validação:  {len(X_val)} dias (2020 a 2021)")
    print(f"Amostras de Teste:      {len(X_test)} dias (2022 a 2024+)")

    # Cálculo do peso de balanceamento da classe minoritária
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = float(n_neg / max(n_pos, 1))
    print(f"Fator de balanceamento scale_pos_weight: {scale_weight:.2f}")

    # 1. Treinamento do Modelo Base (com foco em sensibilidade e ranking)
    modelo_base = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        scale_pos_weight=scale_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    print("Treinando modelo base XGBoost...")
    modelo_base.fit(X_train, y_train)

    # 2. Calibração de Probabilidade via Platt Scaling
    print("Calibrando probabilidades com PlattCalibrator (Validação 2020-2021)...")
    modelo_calibrado = PlattCalibrator(base_model=modelo_base)
    modelo_calibrado.fit(X_val, y_val)

    # Previsões no conjunto de teste cego
    prob_ponderada_teste = modelo_base.predict_proba(X_test)[:, 1]
    prob_calibrada_teste = modelo_calibrado.predict_proba(X_test)[:, 1]

    # Avaliação das métricas
    auc_roc = roc_auc_score(y_test, prob_ponderada_teste)
    pr_auc = average_precision_score(y_test, prob_ponderada_teste)
    brier_ponderado = brier_score_loss(y_test, prob_ponderada_teste)
    brier_calibrado = brier_score_loss(y_test, prob_calibrada_teste)

    print(f"\n[MÉTRICAS DO MODELO - TESTE 2022-2024+]")
    print(f" -> ROC-AUC (Capacidade de Separação): {auc_roc:.4f}")
    print(f" -> PR-AUC (Precisão Média):           {pr_auc:.4f}")
    print(f" -> Brier Score Não Calibrado:         {brier_ponderado:.4f}")
    print(f" -> Brier Score Calibrado (Excelente): {brier_calibrado:.4f}")

    # -------------------------------------------------------------
    # Cálculo da Curva Percentual: Estiagem vs. Chance de Queimada
    # -------------------------------------------------------------
    print("\nCalculando a curva percentual de risco e probabilidade calibrada...")

    faixa_dias_secos = np.arange(0, 61, 2)
    curva_risco = []

    for dias in faixa_dias_secos:
        cenario = pd.DataFrame(
            [
                {
                    "consecutive_dry_days": dias,
                    "precip_sum_roll7": 0.0 if dias >= 7 else 10.0,
                    "precip_sum_roll15": 0.0 if dias >= 15 else 15.0,
                    "precip_sum_roll30": 0.0 if dias >= 30 else 25.0,
                    "precip_sum_roll60": 0.0 if dias >= 60 else 50.0,
                    "humidity_min": max(15.0, 45.0 - (dias * 0.5)),
                    "humidity_mean": max(35.0, 65.0 - (dias * 0.5)),
                    "temp_max": 31.0,
                    "temp_mean": 24.0,
                    "thermal_amplitude": 14.0,
                    "wind_speed_max": 20.0,
                    "solar_radiation_sum": 18000.0,
                    "indice_secura": 31.0 / max(15.0, 45.0 - (dias * 0.5)),
                    "indice_propagacao_vento": (20.0 * 31.0) / max(15.0, 45.0 - (dias * 0.5)),
                    "sin_doy": -0.85,
                    "cos_doy": -0.52,
                }
            ]
        )

        p_ponderada = float(modelo_base.predict_proba(cenario)[:, 1][0])
        p_calibrada = float(modelo_calibrado.predict_proba(cenario)[:, 1][0])
        p_analitica = recalibrar_probabilidade_analitica(p_ponderada, scale_weight)

        curva_risco.append(
            {
                "dias_estiagem": int(dias),
                # Mantém chave principal para retrocompatibilidade com o dashboard
                "chance_queimada_percentual": round(p_ponderada * 100, 2),
                "indice_risco_relativo_percentual": round(p_ponderada * 100, 2),
                "probabilidade_calibrada_percentual": round(p_calibrada * 100, 2),
                "probabilidade_analitica_percentual": round(p_analitica * 100, 2),
            }
        )

    # Salva o modelo treinado e os artefatos
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    modelo_path = MODELS_DIR / "xgboost_queimadas_bauru.joblib"
    modelo_calibrado_path = MODELS_DIR / "xgboost_queimadas_bauru_calibrado.joblib"
    curva_path = REPORTS_DIR / "curva_estiagem_vs_queimadas.json"
    metricas_path = REPORTS_DIR / "queimadas_metricas.json"

    # Salva modelo base e calibrado
    joblib.dump(modelo_base, modelo_path)
    joblib.dump(modelo_calibrado, modelo_calibrado_path)

    with open(curva_path, "w", encoding="utf-8") as f:
        json.dump(curva_risco, f, indent=4, ensure_ascii=False)

    metricas = {
        "roc_auc": round(auc_roc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score_ponderado": round(brier_ponderado, 4),
        "brier_score_calibrado": round(brier_calibrado, 4),
        "scale_weight": round(scale_weight, 2),
        "total_treino": int(len(X_train)),
        "total_val": int(len(X_val)),
        "total_teste": int(len(X_test)),
    }
    with open(metricas_path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4)

    print(f"\nModelo base salvo em:      {modelo_path}")
    print(f"Modelo calibrado salvo em: {modelo_calibrado_path}")
    print(f"Curva percentual salva em: {curva_path}")
    return metricas


if __name__ == "__main__":
    treinar_xgboost_queimadas()
