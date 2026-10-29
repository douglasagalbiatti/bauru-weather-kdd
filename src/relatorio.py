"""
Gera reports/relatorio_final_kdd.md a partir das tabelas de resultados.

Todos os números do relatório vêm dos arquivos de reports/tables/ e dos metadados da
coleta. Assim o texto nunca fica desatualizado em relação ao que o pipeline calculou:
basta rodar `python src/relatorio.py` (ou o reprocessamento completo).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "reports" / "tables"
META = ROOT / "data" / "raw" / "bauru_weather_hourly_2010-01-01_2024-12-31_metadata.json"
OUT = ROOT / "reports" / "relatorio_final_kdd.md"

MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def _load(nome: str) -> Dict[str, Any]:
    with open(TABLES / nome, encoding="utf-8") as f:
        return json.load(f)


def n(x: float, casas: int = 2) -> str:
    """Número no formato brasileiro (vírgula decimal, ponto de milhar)."""
    s = f"{x:,.{casas}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def p(x: float) -> str:
    """p-valor legível."""
    if x < 0.001:
        mant, exp = f"{x:.2e}".split("e")
        return f"{mant.replace('.', ',')} × 10^{int(exp)}"
    return n(x, 3)


def gerar_relatorio() -> Path:
    st = _load("statistical_tests.json")
    ml = _load("ml_evaluation_metrics.json")
    qm = _load("queimadas_metricas.json")
    pre = _load("preprocessing_summary.json")
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}

    cs = st["climate_summary"]
    sub = st["subperiod_comparison"]
    ev = sub["dry_spell_events"]
    e1, e2 = ev["period_1"], ev["period_2"]
    temp = sub["temperature_mean_c"]
    trends = {t["variable"]: t for t in st["mann_kendall_trends"]}
    statn = {s["variable"]: s for s in st["stationarity_tests"]}
    corr = cs["pearson_correlations"]

    tn = cs["monthly_temp_normals_c"]
    pn = cs["monthly_precip_normals_mm"]
    amp = cs["monthly_amplitude_c"]
    mes_quente = max(tn, key=lambda k: tn[k])
    mes_frio = min(tn, key=lambda k: tn[k])
    annual_max = cs["annual_max_dry_spell_days"]
    ano_recorde = max(annual_max, key=lambda k: annual_max[k])

    def linha_trend(nome: str) -> str:
        t = trends[nome]
        unid = "dias/ano" if "Estiagem" in nome else ("mm/ano" if "Precipitação" in nome else "°C/ano")
        conclusao = "Tendência significativa" if t["has_statistically_significant_trend"] else "Sem tendência significativa"
        return f"| {nome} | {t['test']} | {n(t['z_score'])} | {p(t['hypothesis_p_value'])} | {n(t['sens_slope'], 4)} {unid} | {conclusao} |"

    def linha_stat(nome: str) -> str:
        s = statn[nome]
        kp = s["kpss"]
        kp_txt = f"≥ {n(kp['p_value'], 2)}" if kp.get("p_value_is_lower_bound") else p(kp["p_value"])
        adf_ok, kpss_ok = s["adf"]["is_stationary_p05"], kp["is_stationary_p05"]
        leitura = "Estacionária (os dois testes concordam)" if adf_ok and kpss_ok else (
            "Não estacionária (os dois testes concordam)" if not adf_ok and not kpss_ok else "Inconclusiva (os testes discordam)")
        return f"| {nome} | {n(s['adf']['test_statistic'])} | {p(s['adf']['p_value'])} | {n(kp['test_statistic'], 3)} | {kp_txt} | {leitura} |"

    est_dif = "é" if ev["statistically_different"] else "não é"
    temp_dif = "é" if temp["statistically_different"] else "não é"
    bss = qm.get("brier_skill_score")
    feats = ml.get("top_features", [])[:5]
    feats_txt = "\n".join(f"  {i + 1}. `{f['feature']}` ({n(f['importance_gain_pct'], 1)}% do ganho)" for i, f in enumerate(feats))

    md = f"""# Relatório Final – Descoberta de Conhecimento em Séries Temporais Meteorológicas (Bauru/SP)

**Metodologia:** KDD (*Knowledge Discovery in Databases*)
**Município:** Bauru / SP – Brasil
**Coordenadas solicitadas:** latitude `{meta.get('requested_latitude', -22.3147)}`, longitude `{meta.get('requested_longitude', -49.0608)}`
**Ponto de grade ERA5:** latitude `{n(meta.get('grid_latitude', 0), 3)}°`, longitude `{n(meta.get('grid_longitude', 0), 3)}°`, altitude `{n(meta.get('grid_elevation_m', 0), 0)} m`
**Janela do estudo:** {cs['study_years'].replace('-', ' a ')} ({n(cs['n_days'], 0)} dias, {cs['n_months']} meses) — a base continua sendo atualizada diariamente, mas a inferência estatística usa esta janela fixa
**Referência temporal:** horário local padrão (UTC−03:00)
**Fonte primária:** Open-Meteo Historical Weather API (reanálise ERA5 / ERA5-Land)

> Relatório gerado automaticamente por `src/relatorio.py` em {datetime.now():%d/%m/%Y %H:%M}, a partir de `reports/tables/`.

---

## 1. Introdução

Este projeto aplica o ciclo **KDD** para caracterizar o clima de Bauru/SP, testar se há tendências de aquecimento ou de mudança no regime de chuvas e estiagens, e construir modelos de apoio à decisão para calor extremo e risco de queimada.

Variáveis horárias de superfície: `temperature_2m` (°C), `relative_humidity_2m` (%), `precipitation` (mm), `wind_speed_10m` (km/h), `surface_pressure` (hPa) e `shortwave_radiation` (W/m²).

---

## 2. Síntese do Ciclo KDD

```mermaid
graph TD
    A["1. Domínio e objetivos"] --> B["2. Seleção (Open-Meteo, INPE, IPMet)"]
    B --> C["3. Pré-processamento"]
    C --> D["4. Transformação e engenharia de variáveis"]
    D --> E["5. Mineração estatística e modelos"]
    E --> F["6. Interpretação e app em produção"]
```

### Seleção — `src/selection.py`
Cliente oficial `openmeteo-requests` com cache em disco e *retry* com *backoff* exponencial. Os instantes da API (segundos Unix em UTC) são convertidos para horário local somando o deslocamento da resposta.

### Pré-processamento — `src/preprocessing.py`
- Une a carga histórica aos lotes incrementais de `data/raw/`.
- **Fuso horário:** a carga histórica antiga estava em UTC; o deslocamento aplicado nesta execução foi de **{n(pre.get('timezone_audit', {}).get('timezone_shift_hours', 0), 0)} h**.
- Continuidade: {pre['temporal_audit']['duplicates_removed']} duplicatas removidas, {pre['temporal_audit']['missing_timestamps_inserted']} horas faltantes inseridas.
- Limites físicos validados e valores ausentes tratados (interpolação temporal limitada a 3 h; chuva ausente = 0 mm).
- Base horária final: **{n(pre['dataset_shape']['rows'], 0)} registros** ({pre['date_range']['start']} a {pre['date_range']['end']}).

### Transformação — `src/transformation.py`
Agregações diárias e mensais com a função certa para cada variável (média, mínimo, máximo, soma), dias consecutivos sem chuva, médias móveis, *lags*, seno/cosseno do dia do ano e anomalias em relação à climatologia fixa de 2010–2024.

---

## 3. Principais Descobertas

### 3.1 Climatologia
- Temperatura média: **{n(cs['temp_mean_c'], 1)} °C**; mês mais quente: {MESES[int(mes_quente) - 1]} ({n(tn[mes_quente], 1)} °C); mais frio: {MESES[int(mes_frio) - 1]} ({n(tn[mes_frio], 1)} °C).
- Chuva média anual: **{n(cs['annual_precip_mean_mm'], 0)} mm**; janeiro ≈ {n(pn['1'] if '1' in pn else pn[1], 0)} mm e julho ≈ {n(pn['7'] if '7' in pn else pn[7], 0)} mm.
- Amplitude térmica diária: {n(amp['1'] if '1' in amp else amp[1], 1)} °C em janeiro e {n(amp['9'] if '9' in amp else amp[9], 1)} °C em setembro.
- Recordes diários: máxima de {n(cs['temp_max_record']['value_c'], 1)} °C ({cs['temp_max_record']['date']}) e mínima de {n(cs['temp_min_record']['value_c'], 1)} °C ({cs['temp_min_record']['date']}).

![Climograma de Bauru](figures/fig01_climograma_bauru.png)

### 3.2 Estacionariedade (ADF e KPSS)
ADF: H0 = não estacionária. KPSS: H0 = estacionária. Usados juntos porque as hipóteses nulas são opostas.

| Série | ADF (estatística) | ADF (p) | KPSS (estatística) | KPSS (p) | Leitura |
|---|---|---|---|---|---|
{linha_stat("Temperatura Média Diária")}
{linha_stat("Precipitação Diária")}
{linha_stat("Temperatura Média Mensal")}
{linha_stat("Precipitação Mensal")}

### 3.3 Tendências (Mann-Kendall e declive de Sen)
Séries mensais usam o **Mann-Kendall sazonal**, que compara cada mês com o mesmo mês dos outros anos; o teste original não é adequado para séries com sazonalidade forte.

| Série | Teste | Z | p-valor | Sen | Conclusão (α = 5%) |
|---|---|---|---|---|---|
{linha_trend("Temperatura Média Mensal")}
{linha_trend("Temperatura Máxima Mensal")}
{linha_trend("Temperatura Mínima Mensal")}
{linha_trend("Precipitação Total Mensal")}
{linha_trend("Maior Estiagem de Cada Ano")}

![Decomposição STL](figures/fig05_decomposicao_stl_temperatura.png)

### 3.4 Comparação entre subperíodos ({sub['period_1_years']} × {sub['period_2_years']})

**Temperatura (médias anuais):** {n(temp['period_1'])} °C × {n(temp['period_2'])} °C (diferença de {n(temp['difference'])} °C; Welch p = {p(temp['ttest_p_value'])}; Mann-Whitney p = {p(temp['mann_whitney_p_value'])}). A diferença {temp_dif} estatisticamente significativa.

**Estiagens (por evento):** cada sequência ininterrupta de dias sem chuva é uma observação.

| Métrica | {e1['years']} | {e2['years']} |
|---|---|---|
| Eventos de estiagem | {e1['n_events']} ({n(e1['events_per_year'], 1)}/ano) | {e2['n_events']} ({n(e2['events_per_year'], 1)}/ano) |
| Duração média | {n(e1['mean_duration_days'], 1)} dias | {n(e2['mean_duration_days'], 1)} dias |
| Duração mediana | {n(e1['median_duration_days'], 0)} dias | {n(e2['median_duration_days'], 0)} dias |
| Maior estiagem | {e1['max_duration_days']} dias | {e2['max_duration_days']} dias |
| Eventos com 10 dias ou mais | {e1['n_events_10_days_or_more']} | {e2['n_events_10_days_or_more']} |
| Eventos com 30 dias ou mais | {e1['n_events_30_days_or_more']} | {e2['n_events_30_days_or_more']} |

Mann-Whitney sobre as durações: **p = {p(ev['mann_whitney_duration_p_value'])}** — a diferença {est_dif} estatisticamente significativa. A maior estiagem do estudo ocorreu em **{ano_recorde} ({annual_max[ano_recorde]} dias)**.

> **Correção metodológica (29/10/2026).** A versão anterior deste relatório comparou o contador diário de dias secos com Mann-Whitney e obteve p ≈ 1,4 × 10^-8, apresentado como “agravamento comprovado”. O contador não é independente: uma estiagem de 40 dias gera 40 valores (1, 2, …, 40), o que infla artificialmente a significância. A comparação correta é por evento, como na tabela acima.

![Eventos de Estiagem](figures/fig06_eventos_estiagem_dias_secos.png)

### 3.5 Correlações físicas (escala diária, Pearson)
- Temperatura máxima × radiação solar: **r = {n(corr['temp_max_x_solar_radiation_sum'])}**
- Chuva × amplitude térmica: **r = {n(corr['precipitation_sum_x_thermal_amplitude'])}** (nuvens reduzem o aquecimento diurno)
- Temperatura média × pressão: **r = {n(corr['temp_mean_x_surface_pressure_mean'])}** (massas de ar frio vêm com alta pressão)
- Umidade média × amplitude térmica: **r = {n(corr['humidity_mean_x_thermal_amplitude'])}**

![Matriz de Correlações](figures/fig04_heatmap_correlacoes.png)

---

## 4. Modelo 1 — Temperatura máxima de amanhã (XGBoost Regressor)

- **Alvo:** temperatura máxima do dia seguinte (t+1); 23 variáveis do dia t (condições, médias móveis, *lags*, ciclo anual).
- **Divisão cronológica:** treino {ml['train_period']}; teste cego {ml['test_period']}.
- **Baseline:** persistência (amanhã = hoje).

| Métrica | XGBoost | Persistência | Ganho |
|---|---|---|---|
| MAE | {n(ml['xgboost']['mae_celsius'])} °C | {n(ml['baseline_persistence']['mae_celsius'])} °C | {n(ml['improvement_over_baseline_pct'], 1)}% |
| RMSE | {n(ml['xgboost']['rmse_celsius'])} °C | {n(ml['baseline_persistence']['rmse_celsius'])} °C | {n(ml.get('rmse_improvement_over_baseline_pct', 0), 1)}% |
| R² | {n(ml['xgboost']['r2_score'], 3)} | — | — |

Variáveis mais importantes (ganho):
{feats_txt}

**Leitura:** o ganho sobre a persistência é real, porém modesto; o modelo demonstra o método e não substitui uma previsão numérica do tempo.

---

## 5. Modelo 2 — Risco de queimada amanhã (XGBoost Classifier)

- **Alvo:** {qm.get('alvo', 'foco de queimada no dia seguinte (t+1)')}, a partir das condições do dia t.
- **Classe rara:** prevalência de {n(qm.get('prevalencia_treino', 0) * 100, 1)}% no treino ({qm.get('positivos_treino', '?')} dias positivos); `scale_pos_weight` = {n(qm['scale_weight'], 2)}.
- **Divisão:** treino até 2019 ({qm['total_treino']} dias), validação 2020–2021 ({qm['total_val']}), teste {qm.get('periodo_teste', '2022+')} ({qm['total_teste']} dias, {qm.get('positivos_teste', '?')} positivos).
- **Calibração:** Platt scaling ajustado na validação.

| Métrica (teste) | Valor | Referência |
|---|---|---|
| ROC-AUC | {n(qm['roc_auc'], 3)} | 0,5 = acaso |
| PR-AUC | {n(qm['pr_auc'], 3)} | {n(qm.get('prevalencia_teste', 0), 3)} = acaso (prevalência) |
| Brier (calibrado) | {n(qm['brier_score_calibrado'], 4)} | {n(qm.get('brier_score_climatologia', 0), 4)} = sempre prever a taxa-base |
| Brier Skill Score | {n(bss, 3) if bss is not None else '—'} | > 0 = melhor que a taxa-base |

**Leitura:** o modelo é útil para **ordenar** os dias por risco (índice relativo), mas as probabilidades calibradas variam pouco em torno da taxa-base. No app, o escore do modelo ponderado é apresentado como **índice de risco relativo**, e não como probabilidade.

---

## 6. Limitações

1. **Reanálise, não medição:** grade de ~9–25 km; não representa um sensor pontual nem ilhas de calor urbanas.
2. **Horizonte de 15 anos:** curto para tendências climáticas (a OMM usa normais de 30 anos).
3. **Rótulo de queimada ruidoso:** a ausência de foco detectado não garante ausência de fogo (nuvens, resolução e horário de passagem do satélite).
4. **Dados ao vivo no app:** a previsão de amanhã usa previsão numérica ou o sensor do IPMet como entrada, e não a reanálise usada no treino.

## 7. Trabalhos Futuros

1. Correção de viés da reanálise com o histórico da estação IPMet/UNESP (ex.: *quantile mapping*).
2. Validação cruzada temporal e SHAP no regressor; comparação com a previsão da Open-Meteo para o dia seguinte.
3. Modelo de queimada com janelas de vários dias e área ampliada (município + vizinhos), para mais eventos positivos.
4. Estender a série para 30 anos ou mais (ERA5 desde 1940).
"""
    OUT.write_text(md, encoding="utf-8")
    print(f"-> Relatório final gerado em: {OUT}")
    return OUT


if __name__ == "__main__":
    gerar_relatorio()
