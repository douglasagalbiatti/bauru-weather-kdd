"""
Captura e interpretação da estação meteorológica de superfície do IPMet/UNESP Bauru
(https://www.ipmetradar.com.br/2estacao.php).

Módulo compartilhado entre:
- src/update_ipmet.py  -> coletor executado numa máquina com IP brasileiro (self-hosted runner),
                          que grava data/interim/ipmet_latest.json no repositório;
- app.py               -> painel Streamlit, que lê esse snapshot (o site do IPMet não responde
                          a IPs de servidores estrangeiros como os do Streamlit Community Cloud).

Depende apenas de `requests` e da biblioteca padrão, para rodar leve no runner.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

IPMET_URLS = [
    "https://www.ipmetradar.com.br/2estacao.php",
    "http://www.ipmetradar.com.br/2estacao.php",
]

SNAPSHOT_PATH = Path("data/interim/ipmet_latest.json")
HISTORY_PATH = Path("data/raw/ipmet/ipmet_readings.csv")

# Fuso das leituras publicadas pelo IPMet (GMT-3)
BRT = timezone(timedelta(hours=-3))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.ipmetradar.com.br/",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
}

# Campos numéricos extraídos da tabela da estação: nome -> expressão regular
_FIELD_PATTERNS = {
    "temp": r"Temp:\s*([\d\.\-]+)\s*o\s*C",
    "temp_max": r"Max:\s*([\d\.\-]+)\s*o\s*C",
    "temp_min": r"Min:\s*([\d\.\-]+)\s*o\s*C",
    "precipitation_today": r"Hoje:\s*([\d\.\-]+)\s*mm",
    "humidity": r"Umidade:\s*([\d\.\-]+)%",
    "humidity_min": r"Min:\s*([\d\.\-]+)%",
    "wind_speed": r"Veloc:\s*([\d\.\-]+)\s*km/h",
    "wind_gust_max": r"Max dia:\s*([\d\.\-]+)\s*km/h",
    "pressure": r"Pressão:\s*([\d\.\-]+)\s*mB",
    "solar_radiation": r"Radiação Global:\s*([\d\.\-]+)\s*W/m",
}

FIELDS = list(_FIELD_PATTERNS.keys())


def fetch_ipmet_html(timeout: float = 20.0, proxies: Optional[Dict[str, str]] = None) -> str:
    """Baixa a página da estação. Levanta ValueError se nenhuma URL devolver a tabela."""
    session = requests.Session()
    session.headers.update(HEADERS)
    if proxies:
        session.proxies.update(proxies)

    last_err = None
    # allow_redirects=False evita loops de redirecionamento (TooManyRedirects)
    for url in IPMET_URLS:
        try:
            resp = session.get(url, timeout=timeout, allow_redirects=False)
            if resp.status_code == 200 and "<table" in resp.text.lower():
                return resp.text
            if resp.status_code in (301, 302, 303, 307, 308):
                last_err = f"redirecionamento HTTP {resp.status_code} (comum para IPs fora do Brasil)"
            else:
                last_err = f"código HTTP inesperado ({resp.status_code})"
        except Exception as ex:  # noqa: BLE001
            last_err = str(ex)

    raise ValueError(f"Estação IPMet indisponível: {last_err}")


def _to_float(match: Optional[re.Match]) -> Optional[float]:
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_ipmet_html(page_html: str) -> Dict[str, Any]:
    """
    Extrai as leituras da primeira tabela da página.
    Campos ausentes ficam como None (sem valores inventados).
    """
    tables = re.findall(r"<table[^>]*>(.*?)</table>", page_html, re.DOTALL | re.IGNORECASE)
    if not tables:
        raise ValueError("Tabela da estação IPMet não encontrada na página.")

    clean_text = html.unescape(re.sub(r"<[^>]+>", " ", tables[0]))
    clean_text = " ".join(clean_text.split())

    readings: Dict[str, Any] = {
        name: _to_float(re.search(pattern, clean_text)) for name, pattern in _FIELD_PATTERNS.items()
    }

    time_m = re.search(r"Horário da Leitura \(GMT-3\)\s*([\d\/]+ - [\d:]+)", clean_text)
    reading_time_raw = time_m.group(1) if time_m else None
    readings["reading_time_raw"] = reading_time_raw
    parsed = parse_reading_time(reading_time_raw) if reading_time_raw else None
    readings["reading_time"] = parsed.isoformat() if parsed else None

    return readings


def parse_reading_time(raw: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """
    Converte o horário publicado pelo IPMet em datetime (GMT-3).
    Aceita "29/10/2026 - 01:05" e também o formato sem ano usado pelo site, "29/10 - 01:05";
    nesse caso usa o ano atual, ou o anterior se a data cair no futuro (virada de ano).
    """
    raw = raw.strip()
    for fmt in ("%d/%m/%Y - %H:%M:%S", "%d/%m/%Y - %H:%M", "%d/%m/%y - %H:%M"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=BRT)
        except ValueError:
            continue

    now = (now or datetime.now(timezone.utc)).astimezone(BRT)
    for fmt in ("%d/%m - %H:%M:%S", "%d/%m - %H:%M"):
        try:
            # Ano incluído antes do parse para aceitar 29/02 em anos bissextos
            ts = datetime.strptime(f"{now.year}/{raw}", f"%Y/{fmt}").replace(tzinfo=BRT)
        except ValueError:
            continue
        if ts - now > timedelta(days=1):  # ex.: leitura de 31/12 lida em 01/01
            ts = ts.replace(year=now.year - 1)
        return ts

    return None


def load_snapshot(path: Path = SNAPSHOT_PATH) -> Optional[Dict[str, Any]]:
    """Lê o snapshot gravado pelo coletor, ou None se não existir/for inválido."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def snapshot_age(snapshot: Dict[str, Any], now: Optional[datetime] = None) -> Optional[timedelta]:
    """Idade da leitura (pelo horário da estação; se ausente, pelo horário da coleta)."""
    stamp = snapshot.get("reading_time") or snapshot.get("fetched_at_utc")
    if not stamp:
        return None
    try:
        ts = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=BRT)
    now = now or datetime.now(timezone.utc)
    return now - ts
