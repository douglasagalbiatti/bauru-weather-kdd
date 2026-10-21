"""
Módulo de Atualização Incremental de Queimadas (KDD) – Bauru/SP
Verifica a data mais recente na base local de queimadas, busca novos focos
disponíveis no INPE para o município de Bauru e reexecuta o pipeline de fusão.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Dict, Any, Optional
import urllib.request
import zipfile

import pandas as pd

# Garante acesso aos módulos da pasta src
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.queimadas_selection import (
    RAW_QUEIMADAS_DIR,
    INTERIM_DIR,
    filtrar_focos_municipio_bauru,
)
from src.queimadas_transformation import executar_pipeline_transformacao


def get_latest_queimadas_date(
    interim_path: Path = INTERIM_DIR / "bauru_queimadas_diario.parquet",
) -> Optional[pd.Timestamp]:
    """
    Retorna a última data registrada na base diária de queimadas de Bauru.
    """
    if not interim_path.exists():
        return None
    df = pd.read_parquet(interim_path)
    if df.empty:
        return None
    return pd.to_datetime(df["data"]).max()


def update_inpe_ano_corrente(ano: int = None) -> Optional[Path]:
    """
    Baixa ou atualiza o arquivo consolidado do ano corrente do satélite de referência do INPE.
    Diferente de anos passados (estáticos), o ano corrente é atualizado periodicamente pelo INPE.
    """
    if ano is None:
        ano = datetime.now().year

    RAW_QUEIMADAS_DIR.mkdir(parents=True, exist_ok=True)
    base_url = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/anual/Brasil_sat_ref"
    zip_name = f"focos_br_ref_{ano}.zip"
    csv_name = f"focos_br_ref_{ano}.csv"

    zip_path = RAW_QUEIMADAS_DIR / zip_name
    csv_path = RAW_QUEIMADAS_DIR / csv_name

    url = f"{base_url}/{zip_name}"
    print(f"Verificando atualizações de focos de queimada para {ano} no INPE: {url}...")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as f_out:
            f_out.write(resp.read())

        with zipfile.ZipFile(zip_path, "r") as z:
            membros_csv = [m for m in z.namelist() if m.endswith(".csv")]
            for membro in membros_csv:
                nome_final = Path(membro).name
                with z.open(membro) as fonte, open(RAW_QUEIMADAS_DIR / nome_final, "wb") as destino:
                    destino.write(fonte.read())

        zip_path.unlink(missing_ok=True)
        print(f"Sucesso: {csv_name} atualizado.")
        return csv_path
    except Exception as e:
        print(f"Arquivo anual de {ano} ainda não disponível ou erro no download: {e}")
        return None


def executar_atualizacao_queimadas() -> Dict[str, Any]:
    """
    Orquestra o processo de atualização:
    1. Verifica o último registro local.
    2. Baixa dados do ano corrente.
    3. Filtra Bauru e integra na base diária.
    4. Executa a fusão climática com o pipeline de transformação.
    """
    interim_parquet = INTERIM_DIR / "bauru_queimadas_diario.parquet"
    ultima_data = get_latest_queimadas_date(interim_parquet)
    print(f"Última data de queimadas registrada em Bauru: {ultima_data}")

    ano_atual = datetime.now().year
    csv_ano = update_inpe_ano_corrente(ano_atual)

    if csv_ano and csv_ano.exists():
        df_bauru_novo = filtrar_focos_municipio_bauru(csv_ano)

        if not df_bauru_novo.empty:
            # Agrupa novos focos por dia
            df_novo_diario = (
                df_bauru_novo.groupby("data").size().reset_index(name="queimadas_focos_count")
            )
            df_novo_diario["data"] = pd.to_datetime(df_novo_diario["data"])

            if interim_parquet.exists():
                df_antigo = pd.read_parquet(interim_parquet)
                df_antigo["data"] = pd.to_datetime(df_antigo["data"])

                # Unifica e remove duplicatas mantendo a versão mais recente
                df_combinado = pd.concat([df_antigo, df_novo_diario], ignore_index=True)
                df_combinado = df_combinado.drop_duplicates(subset=["data"], keep="last")
                df_combinado = df_combinado.sort_values("data").reset_index(drop=True)
            else:
                df_combinado = df_novo_diario

            df_combinado.to_parquet(interim_parquet, index=False)
            print(f"Base intermediária atualizada com sucesso em: {interim_parquet}")
        else:
            print("Nenhum foco novo detectado em Bauru para o período recente.")

    # Executa a transformação e fusão com os dados meteorológicos atualizados
    print("\nExecutando fusão com a série meteorológica...")
    df_final = executar_pipeline_transformacao()

    resultado = {
        "data_execucao": datetime.now(timezone.utc).isoformat(),
        "total_registros_diarios": len(df_final),
        "total_dias_com_fogo": int(df_final["has_fire"].sum()),
        "status": "sucesso",
    }

    log_path = Path("reports/tables/update_queimadas_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=4)

    return resultado


if __name__ == "__main__":
    executar_atualizacao_queimadas()
