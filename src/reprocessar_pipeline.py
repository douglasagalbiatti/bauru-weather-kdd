"""
Reprocessamento completo do projeto (KDD) – Bauru/SP

Refaz, em ordem, todas as etapas a partir dos dados brutos de data/raw/:
  1. Pré-processamento (carga histórica + lotes incrementais, em horário local)
  2. Transformação (bases diária e mensal)
  3. Mineração estatística (janela fixa 2010–2024)
  4. Figuras do relatório
  5. Modelo XGBoost de temperatura máxima (t+1)
  6. Série de queimadas do INPE (datas em horário local), fusão e modelo de risco (t+1)
  7. Relatório final em Markdown, gerado a partir das tabelas de resultados

Uso (na raiz do projeto, com o ambiente virtual ativo):
    python src/reprocessar_pipeline.py

Opções:
    --sem-queimadas   não refaz a série do INPE (usa data/interim/bauru_queimadas_diario.parquet)
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # os módulos usam caminhos relativos à raiz do projeto


def etapa(numero: int, titulo: str) -> float:
    print("\n" + "=" * 72)
    print(f"ETAPA {numero} – {titulo}")
    print("=" * 72)
    return time.time()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reprocessa todo o pipeline KDD.")
    parser.add_argument("--sem-queimadas", action="store_true", help="não refaz a série do INPE")
    args = parser.parse_args()
    inicio = time.time()

    from src.preprocessing import run_preprocessing_pipeline
    from src.transformation import run_transformation_pipeline
    from src.mining import run_mining_pipeline
    from src.visualization import generate_all_visualizations
    from src.modeling import train_xgboost_model
    from src.queimadas_selection import RAW_QUEIMADAS_DIR, consolidar_serie_focos_bauru
    from src.queimadas_transformation import executar_pipeline_transformacao
    from src.queimadas_modeling import treinar_xgboost_queimadas
    from src.relatorio import gerar_relatorio

    etapa(1, "Pré-processamento")
    df_clean, audit = run_preprocessing_pipeline()
    print(f"   Base horária: {len(df_clean)} registros ({df_clean.index.min()} a {df_clean.index.max()})")

    etapa(2, "Transformação")
    run_transformation_pipeline()

    etapa(3, "Mineração estatística")
    run_mining_pipeline()

    etapa(4, "Figuras")
    generate_all_visualizations()

    etapa(5, "Modelo de temperatura máxima (XGBoost)")
    train_xgboost_model()

    etapa(6, "Queimadas (INPE)")
    csvs = sorted(Path(RAW_QUEIMADAS_DIR).glob("focos_br_ref_*.csv")) if Path(RAW_QUEIMADAS_DIR).exists() else []
    if args.sem_queimadas:
        print("   --sem-queimadas: mantendo a série diária de focos existente.")
    elif not csvs:
        print(f"   Nenhum CSV do INPE em {RAW_QUEIMADAS_DIR}/. Baixando os arquivos anuais (pode demorar)...")
        consolidar_serie_focos_bauru(range(2010, datetime.now().year + 1))
    else:
        print(f"   Refazendo a série a partir de {len(csvs)} arquivos anuais (os que faltarem serão baixados)...")
        consolidar_serie_focos_bauru(range(2010, datetime.now().year + 1))
    executar_pipeline_transformacao()
    treinar_xgboost_queimadas()

    etapa(7, "Relatório final")
    gerar_relatorio()

    print(f"\nReprocessamento concluído em {time.time() - inicio:.0f} s.")
    print("Confira as mudanças com `git status` e `git diff --stat` antes do commit.")


if __name__ == "__main__":
    main()
