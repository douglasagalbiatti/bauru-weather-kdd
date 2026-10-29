# Dicionário de Dados e Metadados da Coleta – Bauru/SP

## 1. Metadados da Coleta (Reanálise ERA5/ERA5-Land)

* **Local de Referência:** Bauru, São Paulo, Brasil
* **Coordenadas Solicitadas:** Latitude `-22.3147`, Longitude `-49.0608`
* **Ponto de Grade (Grid ERA5):** Latitude `-22.31986°`, Longitude `-49.04858°`
* **Elevação do Grid:** `521.0 metros`
* **Fuso Horário:** `America/Sao_Paulo` (UTC-03:00 / offset `-10800` segundos). O índice temporal de todas as bases está em **horário local padrão (UTC−03:00)**; a carga histórica antiga, gravada em UTC, é corrigida automaticamente no pré-processamento.
* **Período Coberto:** `2010-01-01 00:00:00` a `2024-12-31 23:00:00` (15 anos)
* **Total de Registros Horários:** 131.496 linhas
* **Endpoint de Coleta:** `https://archive-api.open-meteo.com/v1/archive`
* **Data da Extração:** 11/10/2026 (via cliente oficial `openmeteo-requests`)

---

## 2. Dicionário de Variáveis Meteorológicas

| Variável | Tipo | Unidade | Descrição Física | Limites Válidos |
|---|---|---|---|---|
| `date` | `datetime64[ns]` | ISO 8601 | Timestamp da observação horária, em horário local padrão (UTC−03:00) | 2010 em diante |
| `temperature_2m` | `float64` | °C | Temperatura do ar a 2 metros do solo | -5.0 a 48.0 |
| `relative_humidity_2m` | `float64` | % | Umidade relativa do ar a 2 metros | 0.0 a 100.0 |
| `precipitation` | `float64` | mm | Precipitação líquida horária acumulada | $\ge 0.0$ |
| `wind_speed_10m` | `float64` | km/h | Velocidade horizontal do vento a 10 metros | $\ge 0.0$ |
| `surface_pressure` | `float64` | hPa | Pressão atmosférica na altitude do terreno (521 m) | 850.0 a 1050.0 |
| `shortwave_radiation` | `float64` | W/m² | Fluxo de radiação solar global incidente na superfície | $\ge 0.0$ |

---

## 3. Variáveis Derivadas na Transformação (`bauru_weather_daily`)

| Variável Derivada | Unidade | Descrição |
|---|---|---|
| `temp_mean` | °C | Média diária de temperatura do ar |
| `temp_min` | °C | Temperatura mínima observada no dia |
| `temp_max` | °C | Temperatura máxima observada no dia |
| `thermal_amplitude` | °C | Amplitude térmica diária (`temp_max - temp_min`) |
| `precipitation_sum` | mm | Acumulado diário de chuva |
| `is_rainy_day` | binário | Indicador de dia chuvoso ($1$ se precipitação $\ge 1.0\text{ mm}$, $0$ caso contrário) |
| `consecutive_dry_days` | dias | Contagem contínua de dias consecutivos sem chuva (*dry spell*) |
| `season` | texto | Estação do ano climatológica no Hemisfério Sul |
| `temp_mean_roll7` | °C | Média móvel centrada/deslizante de 7 dias da temperatura |
| `temp_mean_roll30` | °C | Média móvel de 30 dias da temperatura |
| `temp_mean_lag1` | °C | Temperatura média defasada em 1 dia (*lag 1*) |
| `temp_mean_lag7` | °C | Temperatura média defasada em 7 dias (*lag 7*) |

---

## 4. Especificação do Modelo de Machine Learning (XGBoost Regressor)

### 4.1. Definição do Problema Preditivo
* **Tarefa:** Regressão supervisionada de séries temporais.
* **Variável Alvo ($y_{t+1}$):** Temperatura máxima do ar a 2 metros no dia seguinte (`target_temp_max_next_day`).
* **Justificativa Climatológica:** Antecipar ondas de calor diurnas e picos térmicos críticos para a saúde pública e agricultura em Bauru.

### 4.2. Conjuntos de Dados e Protocolo de Avaliação
* **Período de Treinamento:** 01/01/2010 a 31/12/2021 (12 anos – 4.376 amostras válidas).
* **Período de Teste Cego (*Out-of-Sample*):** 01/01/2022 a 31/12/2024 (3 anos – 1.096 amostras).
* **Protocolo:** Divisão cronológica estrita sem embaralhamento temporal (*no data leakage*).

### 4.3. Hiperparâmetros da Arquitetura
| Hiperparâmetro | Valor Configurado | Descrição Física / Algorítmica |
|---|---|---|
| `n_estimators` | 250 | Número de árvores sequenciais de decisão (*boosting rounds*) |
| `learning_rate` | 0.03 | Taxa de aprendizado (*shrinkage*) para convergência suave |
| `max_depth` | 4 | Profundidade máxima de cada árvore (evita *overfitting*) |
| `subsample` | 0.85 | Fração de amostras sorteadas por árvore (regularização estocástica) |
| `colsample_bytree` | 0.85 | Fração de variáveis físicas sorteadas por divisão de nó |
| `random_state` | 42 | Semente para reprodutibilidade estrita |

### 4.4. Métricas
As métricas do teste cego são geradas a cada treino em `reports/tables/ml_evaluation_metrics.json` e reproduzidas em `reports/relatorio_final_kdd.md`. O período de teste é fixo (2022–2024), para que as métricas permaneçam comparáveis enquanto a base cresce.

---

## 5. Especificação do Modelo de Risco de Queimada (XGBoost Classifier)

* **Tarefa:** classificação binária com classe rara (≈ 2% dos dias).
* **Variável Alvo:** `target_fire_next_day` — houve ao menos um foco do INPE em Bauru no dia seguinte (t+1). As variáveis explicativas são do dia t.
* **Datas dos focos:** convertidas de UTC para horário local (UTC−03:00) antes de agregar por dia.
* **Divisão cronológica:** treino até 2019, validação 2020–2021 (calibração de Platt), teste de 2022 até o último ano com focos publicados pelo INPE.
* **Desbalanceamento:** `scale_pos_weight` = negativos ÷ positivos do treino.
* **Saídas:** o escore do modelo ponderado é um **índice de risco relativo (0–100)**, útil para ordenar os dias; a **probabilidade calibrada** é a estimativa realista de foco no dia seguinte.
* **Métricas:** ROC-AUC, PR-AUC (referência: prevalência), Brier calibrado e Brier Skill Score em relação à taxa-base — em `reports/tables/queimadas_metricas.json`.
