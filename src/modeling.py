"""
Módulo de Machine Learning (XGBoost) para Previsão Meteorológica – Bauru/SP
Treina modelo supervisionado de séries temporais para prever a temperatura máxima
do dia seguinte (t+1) com divisão cronológica estrita (Treino: 2010-2021 | Teste: 2022-2024).
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

TRAIN_END_YEAR = 2021
TEST_YEARS = (2022, 2024)


def prepare_features(df_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constrói a matriz de features (lags, médias móveis, ciclo senoidal)
    e define a variável alvo: temp_max do dia seguinte (t+1).
    """
    df = df_daily.copy()

    # Variável alvo: temp_max amanhã
    df["target_temp_max_next_day"] = df["temp_max"].shift(-1)

    # Ciclo senoidal sazonal (dia do ano)
    day_of_year = df.index.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Lags adicionais
    df["temp_max_lag2"] = df["temp_max"].shift(2)
    df["temp_max_lag3"] = df["temp_max"].shift(3)
    df["humidity_lag1"] = df["humidity_mean"].shift(1)
    df["radiation_lag1"] = df["solar_radiation_sum"].shift(1)
    df["pressure_lag1"] = df["surface_pressure_mean"].shift(1)

    # Remove linhas com NaNs geradas pelos shifts
    df_clean = df.dropna().copy()

    feature_cols = [
        "temp_mean",
        "temp_min",
        "temp_max",
        "thermal_amplitude",
        "humidity_mean",
        "humidity_min",
        "precipitation_sum",
        "wind_speed_mean",
        "wind_speed_max",
        "surface_pressure_mean",
        "solar_radiation_sum",
        "temp_mean_roll7",
        "temp_mean_roll30",
        "precip_sum_roll7",
        "temp_mean_lag1",
        "temp_mean_lag7",
        "temp_max_lag2",
        "temp_max_lag3",
        "humidity_lag1",
        "radiation_lag1",
        "pressure_lag1",
        "sin_doy",
        "cos_doy",
    ]

    X = df_clean[feature_cols]
    y = df_clean["target_temp_max_next_day"]
    return X, y


def train_xgboost_model(
    daily_parquet_path: str = "data/processed/bauru_weather_daily.parquet",
    models_dir: str = "models",
    reports_dir: str = "reports/tables",
) -> Dict[str, Any]:
    """
    Executa o treinamento cronológico do XGBoost e salva artefatos e métricas.
    """
    print(f"1. Carregando dados diários de: {daily_parquet_path}")
    df_daily = pd.read_parquet(daily_parquet_path)

    print("2. Engenharia de features temporais e alvo t+1...")
    X, y = prepare_features(df_daily)

    # Divisão Cronológica: Treino 2010–2021 | Teste 2022–2024 (anos completos do estudo).
    # O teste é fixo: com a base crescendo todo dia, "ano >= 2022" mudaria o conjunto de teste
    # e as métricas deixariam de ser comparáveis com as publicadas.
    train_mask = X.index.year <= TRAIN_END_YEAR
    test_mask = (X.index.year >= TEST_YEARS[0]) & (X.index.year <= TEST_YEARS[1])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"   Treino: {X_train.shape[0]} amostras ({X_train.index.min().year} a {TRAIN_END_YEAR})")
    print(f"   Teste:  {X_test.shape[0]} amostras ({TEST_YEARS[0]} a {TEST_YEARS[1]})")

    # Baseline Ingênuo de Persistência: temperatura máxima de amanhã = temperatura máxima de hoje
    y_baseline_pred = X_test["temp_max"]
    mae_baseline = mean_absolute_error(y_test, y_baseline_pred)
    rmse_baseline = float(np.sqrt(mean_squared_error(y_test, y_baseline_pred)))

    print("3. Treinando o modelo XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("4. Avaliando predições no conjunto de teste fora da amostra...")
    y_pred = model.predict(X_test)

    mae_xgb = mean_absolute_error(y_test, y_pred)
    rmse_xgb = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2_xgb = r2_score(y_test, y_pred)

    # Importância das Features (Gain)
    importance_dict = model.get_booster().get_score(importance_type="gain")
    total_gain = sum(importance_dict.values())
    feat_importance = [
        {"feature": k, "importance_gain_pct": round((v / total_gain) * 100, 2)}
        for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
    ]

    metrics = {
        "model": "XGBoost Regressor (Previsão de Temp. Máxima t+1)",
        "train_period": f"{X_train.index.min().year}-{TRAIN_END_YEAR} ({X_train.shape[0]:,} dias)".replace(",", "."),
        "test_period": f"{TEST_YEARS[0]}-{TEST_YEARS[1]} ({X_test.shape[0]:,} dias)".replace(",", "."),
        "train_years": [int(X_train.index.min().year), TRAIN_END_YEAR],
        "test_years": list(TEST_YEARS),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "xgboost": {
            "mae_celsius": round(float(mae_xgb), 3),
            "rmse_celsius": round(float(rmse_xgb), 3),
            "r2_score": round(float(r2_xgb), 3),
        },
        "baseline_persistence": {
            "mae_celsius": round(float(mae_baseline), 3),
            "rmse_celsius": round(float(rmse_baseline), 3),
        },
        "improvement_over_baseline_pct": round(
            float((mae_baseline - mae_xgb) / mae_baseline * 100), 2
        ),
        "rmse_improvement_over_baseline_pct": round(
            float((rmse_baseline - rmse_xgb) / rmse_baseline * 100), 2
        ),
        "top_features": feat_importance[:10],
    }

    # Persistência
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    model_file = models_path / "xgboost_weather_bauru.joblib"
    joblib.dump(model, model_file)

    metrics_file = reports_path / "ml_evaluation_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Salva predições para plotagem no Streamlit
    df_preds = pd.DataFrame(
        {
            "date": X_test.index,
            "real_temp_max": y_test.values,
            "pred_temp_max_xgb": y_pred,
            "pred_baseline": y_baseline_pred.values,
            "error_celsius": np.abs(y_test.values - y_pred),
        }
    ).set_index("date")

    preds_file = reports_path / "ml_predictions_test.parquet"
    df_preds.to_parquet(preds_file)

    print(f"-> Modelo salvo em: {model_file}")
    print(f"-> Métricas salvas em: {metrics_file}")
    print(f"-> Erro Médio Absoluto (MAE): {mae_xgb:.2f} °C (Melhoria de {metrics['improvement_over_baseline_pct']}% sobre o baseline)")

    return metrics


if __name__ == "__main__":
    train_xgboost_model()
