"""Testes do parser da estação IPMet/UNESP Bauru (sem acesso à rede)."""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ipmet import parse_ipmet_html, snapshot_age  # noqa: E402

PAGINA = """<html><body><table>
<tr><td>Horário da Leitura (GMT-3)</td><td>29/10/2026 - 00:40</td></tr>
<tr><td>Temp: 21.4 o C</td><td>Max: 30.2 o C</td><td>Min: 18.9 o C</td></tr>
<tr><td>Chuva Hoje: 0.0 mm</td><td>Umidade: 78%</td><td>Min: 41%</td></tr>
<tr><td>Veloc: 6.5 km/h</td><td>Max dia: 32.0 km/h</td>
<td>Press&atilde;o: 954.1 mB</td><td>Radia&ccedil;&atilde;o Global: 0 W/m2</td></tr>
</table></body></html>"""


def test_parse_campos_principais():
    r = parse_ipmet_html(PAGINA)
    assert r["temp"] == 21.4
    assert r["temp_max"] == 30.2
    assert r["temp_min"] == 18.9
    assert r["humidity"] == 78.0
    assert r["humidity_min"] == 41.0
    assert r["pressure"] == 954.1
    assert r["reading_time"] == "2026-10-29T00:40:00-03:00"


def test_campo_ausente_vira_none():
    r = parse_ipmet_html("<table><tr><td>Temp: 20.0 o C</td></tr></table>")
    assert r["temp"] == 20.0
    assert r["pressure"] is None
    assert r["reading_time"] is None


def test_idade_do_snapshot():
    r = parse_ipmet_html(PAGINA)
    agora = datetime(2026, 10, 29, 4, 10, tzinfo=timezone.utc)  # 01:10 em Brasília
    assert snapshot_age(r, now=agora).total_seconds() == 30 * 60


def test_horario_sem_ano_usa_ano_atual():
    from src.ipmet import parse_reading_time

    agora = datetime(2026, 10, 29, 4, 10, tzinfo=timezone.utc)
    assert parse_reading_time("29/10 - 01:05", now=agora).isoformat() == "2026-10-29T01:05:00-03:00"


def test_horario_sem_ano_na_virada_do_ano():
    from src.ipmet import parse_reading_time

    agora = datetime(2027, 1, 1, 3, 30, tzinfo=timezone.utc)  # 00:30 de 01/01 em Brasília
    assert parse_reading_time("31/12 - 23:55", now=agora).isoformat() == "2026-12-31T23:55:00-03:00"


if __name__ == "__main__":
    test_parse_campos_principais()
    test_campo_ausente_vira_none()
    test_idade_do_snapshot()
    test_horario_sem_ano_usa_ano_atual()
    test_horario_sem_ano_na_virada_do_ano()
    print("Todos os testes de test_ipmet passaram com sucesso!")
