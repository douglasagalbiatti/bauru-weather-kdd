"""
Módulo de Seleção e Ingestão de Dados de Queimadas (INPE) - Bauru/SP
Responsável por baixar os dados anuais do satélite de referência do INPE,
filtrar exclusivamente os focos do município de Bauru e gerar a série diária de focos.
"""

from pathlib import Path
import urllib.request
import zipfile
import pandas as pd


# Diretórios de armazenamento de dados brutos e intermediários
RAW_QUEIMADAS_DIR = Path("data/raw/queimadas")
INTERIM_DIR = Path("data/interim")


def download_e_extrair_ano_inpe(ano: int, output_dir: Path = RAW_QUEIMADAS_DIR) -> Path:
    """
    Baixa o arquivo ZIP anual do satélite de referência do INPE para o Brasil,
    extrai o CSV e armazena na pasta de dados brutos.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # URL oficial do repositório de dados abertos do INPE (Satélite de Referência)
    base_url = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/anual/Brasil_sat_ref"
    zip_name = f"focos_br_ref_{ano}.zip"
    csv_name = f"focos_br_ref_{ano}.csv"

    csv_path = output_dir / csv_name
    zip_path = output_dir / zip_name

    # Se o CSV já existe no disco, não faz o download novamente (cache inteligente)
    if csv_path.exists():
        print(f"[CACHE] CSV do ano {ano} já existe em {csv_path}")
        return csv_path

    url = f"{base_url}/{zip_name}"
    print(f"Baixando dados de {ano} do INPE: {url}...")

    try:
        # Usamos User-Agent para evitar bloqueio HTTP 403 de servidores de dados públicos
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())

        # Descompactação inteligente do CSV (extrai na raiz de output_dir mesmo se o ZIP contiver subpastas como tmp/)
        with zipfile.ZipFile(zip_path, "r") as z:
            membros_csv = [m for m in z.namelist() if m.endswith(".csv")]
            for membro in membros_csv:
                nome_final = Path(membro).name
                with z.open(membro) as fonte, open(output_dir / nome_final, "wb") as destino:
                    destino.write(fonte.read())

        # Remove o arquivo .zip para não ocupar espaço redundante
        zip_path.unlink(missing_ok=True)
        print(f"Sucesso: {csv_name} extraído.")
        return csv_path
    except Exception as erro:
        print(f"Falha ao baixar ano {ano}: {erro}")
        return None


def filtrar_focos_municipio_bauru(csv_path: Path) -> pd.DataFrame:
    """
    Lê o CSV anual do INPE e filtra exclusivamente os registros de Bauru/SP,
    otimizando a memória ao carregar apenas as colunas necessárias.
    """
    if not csv_path or not csv_path.exists():
        return pd.DataFrame()

    try:
        # Lemos primeiro apenas o cabeçalho para validar os nomes das colunas
        df_sample = pd.read_csv(csv_path, nrows=5, sep=None, engine="python")
        cols_presentes = [c for c in df_sample.columns]

        # Mapeamento dinâmico caso os nomes estejam acentuados ou variem
        col_municipio = next((c for c in cols_presentes if "municip" in c.lower()), "municipio")
        col_data = next((c for c in cols_presentes if "data" in c.lower()), "data_hora_gmt")

        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="latin1")

        # Filtra estritamente o município de BAURU (ignorando maiúsculas/minúsculas e espaços)
        df_bauru = df[df[col_municipio].astype(str).str.strip().str.upper() == "BAURU"].copy()

        if df_bauru.empty:
            print(f"Aviso: Nenhum foco registrado em Bauru no arquivo {csv_path.name}")
            return pd.DataFrame()

        # O INPE publica o horário em GMT/UTC. Convertemos para o horário local (UTC-03:00)
        # antes de extrair a data, no mesmo referencial da série meteorológica: um foco
        # detectado às 22h locais pertence ao próprio dia, e não ao dia seguinte em UTC.
        horario_utc = pd.to_datetime(df_bauru[col_data], utc=True)
        df_bauru["data"] = (horario_utc - pd.Timedelta(hours=3)).dt.tz_localize(None).dt.date
        print(f"Ano {csv_path.name}: {len(df_bauru)} focos encontrados em Bauru.")
        return df_bauru

    except Exception as e:
        print(f"Erro ao processar {csv_path.name}: {e}")
        return pd.DataFrame()


def consolidar_serie_focos_bauru(anos: list = range(2010, 2026)) -> pd.DataFrame:
    """
    Executa o ciclo completo para a lista de anos:
    1. Baixa cada ano.
    2. Filtra Bauru.
    3. Concatena e salva a base agregada em data/interim/bauru_queimadas_diario.parquet.
    """
    lista_focos_bauru = []

    for ano in anos:
        csv_file = download_e_extrair_ano_inpe(ano)
        if csv_file:
            df_ano_bauru = filtrar_focos_municipio_bauru(csv_file)
            if not df_ano_bauru.empty:
                lista_focos_bauru.append(df_ano_bauru)

    if not lista_focos_bauru:
        print("Nenhum dado encontrado para consolidar.")
        return pd.DataFrame()

    df_consolidado = pd.concat(lista_focos_bauru, ignore_index=True)

    # Agrupamento diário: contagem de focos por data em Bauru
    df_diario = df_consolidado.groupby("data").size().reset_index(name="queimadas_focos_count")
    df_diario["data"] = pd.to_datetime(df_diario["data"])
    df_diario = df_diario.sort_values("data").reset_index(drop=True)

    # Salva o resultado intermediário
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    saida_parquet = INTERIM_DIR / "bauru_queimadas_diario.parquet"
    df_diario.to_parquet(saida_parquet, index=False)
    print(f"\nSérie diária de queimadas de Bauru consolidada salva em: {saida_parquet}")
    print(f"Total de dias com focos registrados em Bauru: {len(df_diario)}")

    return df_diario


if __name__ == "__main__":
    print("Iniciando coleta e consolidação dos focos de queimada em Bauru (2010 a 2025)...")
    consolidar_serie_focos_bauru(range(2010, 2026))
