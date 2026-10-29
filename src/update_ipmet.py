"""
Coletor da estação IPMet/UNESP Bauru – executado numa máquina com IP brasileiro
(self-hosted runner do GitHub Actions; ver .github/workflows/update_ipmet.yml).

- Grava a leitura mais recente em data/interim/ipmet_latest.json (lida pelo app.py);
- Acrescenta a leitura ao histórico data/raw/ipmet/ipmet_readings.csv;
- Não altera nada se a estação ainda não publicou leitura nova (evita commits vazios);
- Sai com código 1 se não conseguir capturar dados válidos (o snapshot anterior é mantido).

Uso:  python src/update_ipmet.py
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.ipmet import (  # noqa: E402
    FIELDS,
    HISTORY_PATH,
    SNAPSHOT_PATH,
    fetch_ipmet_html,
    load_snapshot,
    parse_ipmet_html,
)


def main() -> int:
    snapshot_path = root_dir / SNAPSHOT_PATH
    history_path = root_dir / HISTORY_PATH

    try:
        readings = parse_ipmet_html(fetch_ipmet_html())
    except Exception as ex:  # noqa: BLE001
        print(f"ERRO: {ex}")
        return 1

    if readings.get("temp") is None or readings.get("reading_time") is None:
        print(f"ERRO: leitura incompleta (temperatura ou horário ausente): {readings}")
        return 1

    previous = load_snapshot(snapshot_path)
    if previous and previous.get("reading_time") == readings["reading_time"]:
        print(f"Sem leitura nova (última: {readings['reading_time']}). Nada a gravar.")
        return 0

    snapshot = {
        "source": "IPMet/UNESP Bauru – https://www.ipmetradar.com.br/2estacao.php",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **readings,
    }

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
        f.write("\n")

    columns = ["reading_time", "fetched_at_utc", *FIELDS]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not history_path.exists() or history_path.stat().st_size == 0
    with open(history_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(snapshot)

    print(
        f"Leitura gravada: {readings['reading_time']} | temp={readings['temp']} °C | "
        f"umidade={readings['humidity']}% | chuva hoje={readings['precipitation_today']} mm"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
