# Estudo de Séries Temporais Meteorológicas – Bauru/SP (Open-Meteo)

**Metodologia:** KDD (Knowledge Discovery in Databases)
**Local:** Bauru/SP, Brasil (latitude -22.3147, longitude -49.0608)
**Fonte:** https://open-meteo.com/en/docs/historical-weather-api
**Endpoint base:** `https://archive-api.open-meteo.com/v1/archive`

---

## 0. Regra de Governança para o Antigravity

> **OBRIGATÓRIO:** o Antigravity **deve voltar (apresentar) o código ao usuário e aguardar aprovação ANTES de implementá-lo no projeto**.
>
> Fluxo exigido para cada etapa:
> 1. Escrever o código proposto e exibi-lo ao usuário (na conversa).
> 2. Explicar explicitamente o que ele faz.
> 3. Aguardar a confirmação explícita do usuário.
> 4. Só então criar/alterar arquivos do projeto.
>
> Nenhum arquivo deve ser criado, editado ou executado sem esse ciclo de aprovação.

---

## 1. Step by Step (KDD)

### Etapa 1 – Compreensão do domínio e definição de objetivos
1. Definir as perguntas de pesquisa (ex.: tendência de temperatura, sazonalidade da chuva, eventos extremos).
2. Definir as variáveis de interesse (ex.: `temperature_2m`, `relative_humidity_2m`, `precipitation`, `wind_speed_10m`, `surface_pressure`, `shortwave_radiation`).
3. Definir o período e a granularidade (horária ou diária).
4. Ler a documentação da API histórica e registrar suas características:
   - Os dados são de **reanálise** (modelos como ERA5/ERA5-Land), **não são medições de uma estação física**. Isso deve constar nas limitações do estudo.
   - Não é necessária API key para uso não comercial, mas há limites de requisições. Consultar a documentação vigente.
   - Existe um pequeno atraso entre a data atual e a última data disponível. 

### Etapa 2 – Seleção (Selection)
1. Definir os parâmetros da requisição: `latitude`, `longitude`, `start_date`, `end_date`, `hourly` e/ou `daily`, `timezone=America/Sao_Paulo`.
2. Optar por consultar por janelas (ex.: ano a ano) se o período for longo, para evitar requisições muito grandes.
3. Usar cache e *retry* para tolerar falhas de rede e evitar requisições repetidas.
4. Converter a resposta em DataFrame e salvar os dados brutos em `data/raw/`, sem alterações.
5. Registrar os metadados da coleta (data da coleta, parâmetros, coordenadas, elevação retornada pela API).

### Etapa 3 – Pré-processamento e limpeza (Preprocessing)
1. Garantir que a coluna de tempo seja `datetime` com fuso `America/Sao_Paulo`.
2. Padronizar nomes de colunas e verificar unidades (a API retorna `°C`, `km/h`, `mm`, `hPa`, `W/m²` por padrão).
3. Checar a estrutura: tipos, dimensões, valores nulos.
4. Remover duplicatas e ordenar pelo índice temporal.
5. Tratar valores ausentes (interpolação temporal, remoção ou manutenção, com justificativa).
6. Validar limites físicos (UR 0–100 %, precipitação ≥ 0, pressão plausível) e detectar *outliers* (IQR, Z-score).
7. Garantir índice temporal contínuo e sem lacunas.
8. Salvar em `data/processed/`.

### Etapa 4 – Transformação (Transformation)
1. Reamostrar a série (horária → diária/mensal/anual) com agregações adequadas: média para temperatura, soma para precipitação, máximo para rajada de vento.
2. Criar variáveis derivadas: ano, mês, estação do ano, hora do dia, amplitude térmica diária, dias chuvosos, dias consecutivos sem chuva.
3. Criar médias móveis, *lags* e diferenças.
4. Calcular anomalias em relação à climatologia (valor − média do mesmo mês).
5. Normalizar/padronizar quando necessário.
6. Decompor a série (tendência, sazonalidade, resíduo).

### Etapa 5 – Mineração de dados / Análise (Data Mining)
1. **Análise exploratória (EDA):** estatísticas descritivas, distribuições, correlações.
2. **Visualização:** séries temporais, histogramas, *boxplots* por mês, *heatmaps* de correlação, rosa dos ventos, climograma.
3. **Séries temporais:** testes de estacionariedade (ADF, KPSS), ACF/PACF, decomposição STL.
4. **Inferência estatística:** testes de hipótese (ex.: comparação entre décadas), intervalos de confiança, tendência com Mann-Kendall.
5. **Modelagem (opcional):** SARIMA, regressão, clusterização de regimes climáticos.
6. Validar com separação temporal treino/teste (sem embaralhar) e métricas (MAE, RMSE, MAPE).

### Etapa 6 – Interpretação e avaliação (Interpretation/Evaluation)
1. Interpretar os resultados à luz das perguntas de pesquisa.
2. Avaliar limitações: dados de reanálise (resolução espacial da grade), possíveis vieses em relação a estações locais, período coberto.
3. Documentar achados e inferências com gráficos claros.
4. Consolidar o resultado em notebook/relatório reprodutível, com conclusões e trabalhos futuros.

---

## 2. Bibliotecas Python por Etapa do KDD

| Etapa KDD | Biblioteca | Finalidade |
|---|---|---|
| Seleção | `openmeteo-requests` | Cliente oficial da Open-Meteo (resposta eficiente em formato binário) |
| Seleção | `requests-cache` | Cache das requisições HTTP |
| Seleção | `retry-requests` | Novas tentativas automáticas em caso de falha |
| Seleção | `requests` | Alternativa: consumo direto da API em JSON |
| Seleção/Pré-proc. | `pathlib`, `json`, `datetime` (stdlib) | Caminhos, arquivos e datas |
| Pré-proc./Transf. | `pandas` | Manipulação tabular e séries temporais |
| Pré-proc./Transf. | `numpy` | Operações numéricas |
| Pré-proc. | `missingno` | Visualização de dados ausentes |
| Mineração | `scipy` | Testes estatísticos |
| Mineração | `statsmodels` | ADF, KPSS, ACF/PACF, decomposição, SARIMA |
| Mineração | `pymannkendall` | Teste de tendência Mann-Kendall |
| Mineração | `scikit-learn` | Escalonamento, clusterização, regressão, métricas |
| Visualização | `matplotlib` | Gráficos base |
| Visualização | `seaborn` | Gráficos estatísticos |
| Visualização | `plotly` | Gráficos interativos |
| Visualização | `windrose` | Rosa dos ventos |
| Ambiente | `jupyter` | Notebooks |

Instalação sugerida:

```bash
pip install openmeteo-requests requests-cache retry-requests requests pandas numpy missingno scipy statsmodels pymannkendall scikit-learn matplotlib seaborn plotly windrose jupyter
```

---

## 3. Comandos Básicos de Cada Biblioteca

### openmeteo-requests (com requests-cache e retry-requests)
| Comando | O que faz |
|---|---|
| `requests_cache.CachedSession(".cache", expire_after=-1)` | Cria sessão HTTP com cache em disco |
| `retry(cache_session, retries=5, backoff_factor=0.2)` | Adiciona novas tentativas com espera crescente |
| `openmeteo_requests.Client(session=retry_session)` | Cria o cliente da API |
| `client.weather_api(url, params=params)` | Executa a requisição e retorna uma lista de respostas (uma por localização) |
| `response = responses[0]` | Seleciona a resposta da primeira localização |
| `response.Latitude()` / `response.Longitude()` / `response.Elevation()` | Retorna as coordenadas e a elevação da grade usada |
| `response.Hourly()` / `response.Daily()` | Acessa o bloco de dados horários / diários |
| `hourly.Variables(0).ValuesAsNumpy()` | Extrai os valores da variável (na ordem solicitada em `params`) como array NumPy |
| `hourly.Time()`, `hourly.TimeEnd()`, `hourly.Interval()` | Retornam início, fim e intervalo (s) da série, para montar o índice com `pd.date_range` |

Parâmetros típicos de `params`:
```python
params = {
    "latitude": -22.3147,
    "longitude": -49.0608,
    "start_date": "2010-01-01",
    "end_date": "2024-12-31",
    "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"],
    "timezone": "America/Sao_Paulo",
}
```

### requests (alternativa)
| Comando | O que faz |
|---|---|
| `requests.get(url, params=...)` | Envia requisição GET à API |
| `response.raise_for_status()` | Lança exceção se o status HTTP indicar erro |
| `response.json()` | Converte a resposta em dicionário Python |
| `response.status_code` | Retorna o código HTTP |

### pandas
| Comando | O que faz |
|---|---|
| `pd.DataFrame(data=dict)` | Cria DataFrame a partir de dicionário |
| `pd.date_range(start, end, freq="h", inclusive="left")` | Cria o eixo temporal |
| `pd.read_csv(path)` / `pd.read_parquet(path)` | Lê arquivos para DataFrame |
| `pd.json_normalize(data)` | Achata JSON aninhado em tabela |
| `df.head()` / `df.tail()` | Mostra as primeiras/últimas linhas |
| `df.info()` | Tipos de dados e contagem de nulos |
| `df.describe()` | Estatísticas descritivas |
| `pd.to_datetime(col)` | Converte para datetime |
| `.dt.tz_convert("America/Sao_Paulo")` | Converte o fuso horário |
| `df.set_index("date")` | Define índice temporal |
| `df.sort_index()` | Ordena pelo índice |
| `df.rename(columns={...})` | Renomeia colunas |
| `df.drop_duplicates()` | Remove duplicatas |
| `df.isna().sum()` | Conta valores ausentes por coluna |
| `df.interpolate(method="time")` | Interpola valores ausentes no tempo |
| `df.fillna(...)` / `df.dropna()` | Preenche / remove nulos |
| `df.resample("D").agg({...})` | Reamostra com agregação por coluna |
| `df.rolling(window=7).mean()` | Média móvel |
| `df.shift(1)` | Cria *lag* |
| `df.groupby(df.index.month).mean()` | Agrega por mês |
| `df.corr()` | Matriz de correlação |
| `df.to_csv(path)` / `df.to_parquet(path)` | Salva dados |

### numpy
| Comando | O que faz |
|---|---|
| `np.mean`, `np.std`, `np.median` | Média, desvio padrão, mediana |
| `np.percentile(a, [25, 75])` | Quartis (para IQR) |
| `np.where(cond, a, b)` | Seleção condicional vetorizada |
| `np.isnan(x)` | Detecta valores nulos |

### missingno
| Comando | O que faz |
|---|---|
| `msno.matrix(df)` | Mapa visual de dados ausentes |
| `msno.bar(df)` | Completude por coluna |

### scipy.stats
| Comando | O que faz |
|---|---|
| `stats.zscore(x)` | Calcula Z-score (outliers) |
| `stats.shapiro(x)` | Teste de normalidade |
| `stats.ttest_ind(a, b)` | Teste t entre dois grupos |
| `stats.mannwhitneyu(a, b)` | Alternativa não paramétrica |
| `stats.pearsonr(x, y)` / `stats.spearmanr(x, y)` | Correlações |

### statsmodels
| Comando | O que faz |
|---|---|
| `adfuller(series)` | Teste ADF de estacionariedade |
| `kpss(series)` | Teste KPSS de estacionariedade |
| `seasonal_decompose(series, period=...)` | Decomposição clássica |
| `STL(series, period=...).fit()` | Decomposição STL robusta |
| `plot_acf(series)` / `plot_pacf(series)` | Gráficos de autocorrelação |
| `SARIMAX(series, order, seasonal_order).fit()` | Ajusta modelo SARIMA |
| `model.forecast(steps=n)` | Previsão de n passos |

### pymannkendall
| Comando | O que faz |
|---|---|
| `mk.original_test(series)` | Teste de tendência monotônica |
| `mk.seasonal_test(series, period=12)` | Teste com sazonalidade |

### scikit-learn
| Comando | O que faz |
|---|---|
| `StandardScaler().fit_transform(X)` | Padroniza variáveis |
| `KMeans(n_clusters=k).fit(X)` | Clusteriza dados |
| `LinearRegression().fit(X, y)` | Regressão linear |
| `TimeSeriesSplit(n_splits=5)` | Validação cruzada temporal |
| `mean_absolute_error`, `mean_squared_error` | Métricas de erro |

### matplotlib
| Comando | O que faz |
|---|---|
| `plt.subplots(figsize=(...))` | Cria figura e eixos |
| `ax.plot(x, y)` | Gráfico de linha |
| `ax.hist(x, bins=30)` | Histograma |
| `ax.set_title/xlabel/ylabel` | Títulos e rótulos |
| `plt.tight_layout()` | Ajusta espaçamento |
| `fig.savefig("fig.png", dpi=150)` | Salva a figura |

### seaborn
| Comando | O que faz |
|---|---|
| `sns.lineplot(data=df, x=..., y=...)` | Linha de séries temporais |
| `sns.boxplot(data=df, x="mes", y="temp")` | Boxplot por categoria |
| `sns.histplot(df["temp"], kde=True)` | Histograma com densidade |
| `sns.heatmap(df.corr(), annot=True)` | Mapa de calor de correlação |
| `sns.pairplot(df)` | Matriz de dispersão |

### plotly
| Comando | O que faz |
|---|---|
| `px.line(df, x=..., y=...)` | Linha interativa |
| `px.box(df, x=..., y=...)` | Boxplot interativo |
| `fig.update_layout(...)` | Ajusta layout |
| `fig.write_html("fig.html")` | Exporta gráfico interativo |

### windrose
| Comando | O que faz |
|---|---|
| `WindroseAxes.from_ax()` | Cria eixo de rosa dos ventos |
| `ax.bar(direction, speed, normed=True)` | Plota a rosa dos ventos |

---

## 4. Estrutura de Pastas do Projeto

```
bauru-weather-kdd/
├── PROJETO_KDD.md              # Este documento (guia metodológico)
├── data/
│   ├── raw/                    # Etapa 2 – Seleção: dados brutos da Open-Meteo (imutáveis)
│   ├── interim/                # Etapa 3 – Pré-processamento: dados parcialmente tratados
│   └── processed/              # Etapas 3 e 4 – Dados limpos e transformados, prontos para análise
├── notebooks/                  # Exploração e narrativa analítica (um notebook por etapa KDD)
├── src/
│   ├── __init__.py
│   ├── selection.py            # Etapa 2 – Coleta via API
│   ├── preprocessing.py        # Etapa 3 – Limpeza e validação
│   ├── transformation.py       # Etapa 4 – Reamostragem, variáveis derivadas, decomposição
│   ├── mining.py               # Etapa 5 – Estatística, inferência e modelagem
│   └── visualization.py        # Etapas 5 e 6 – Gráficos reutilizáveis
├── reports/
│   ├── figures/                # Gráficos exportados (PNG/HTML)
│   └── tables/                 # Tabelas de resultados (CSV)
└── docs/                       # Documentação complementar e metadados da coleta
```

### Correspondência entre pastas e etapas do KDD

| Etapa KDD | Código | Entrada | Saída |
|---|---|---|---|
| 2 – Seleção | `src/selection.py` | API Open-Meteo | `data/raw/` |
| 3 – Pré-processamento | `src/preprocessing.py` | `data/raw/` | `data/interim/` → `data/processed/` |
| 4 – Transformação | `src/transformation.py` | `data/interim/` | `data/processed/` |
| 5 – Mineração | `src/mining.py`, `src/visualization.py` | `data/processed/` | `reports/figures/`, `reports/tables/` |
| 6 – Interpretação | `notebooks/` | `reports/` | Conclusões e relatório final |

### Regras da estrutura
- `data/raw/` nunca é editada manualmente: qualquer correção ocorre no código de pré-processamento.
- Toda lógica reutilizável fica em `src/`; os notebooks apenas importam e executam essas funções.
- Os arquivos de `src/` são criados **vazios**. Seu conteúdo só entra no projeto após o código ser mostrado ao usuário e aprovado (ver seção 0).
