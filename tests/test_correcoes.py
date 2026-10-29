"""Testes das correções de 29/10/2026: fuso horário da carga histórica e eventos de estiagem."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import ensure_local_time, load_incremental_raws  # noqa: E402
from src.transformation import extract_dry_spell_events  # noqa: E402


def _meta(tmp_path, **extra):
    meta = {"start_date": "2010-01-01", "timezone_offset_seconds": -10800, **extra}
    caminho = tmp_path / "meta.json"
    caminho.write_text(json.dumps(meta), encoding="utf-8")
    return str(caminho)


def test_carga_legada_em_utc_e_deslocada_para_horario_local(tmp_path):
    idx = pd.date_range("2010-01-01 03:00", periods=48, freq="h")
    df = pd.DataFrame({"temperature_2m": range(48)}, index=idx)
    corrigido, audit = ensure_local_time(df, _meta(tmp_path))
    assert audit["legacy_utc_detected"] is True
    assert audit["timezone_shift_hours"] == -3
    assert corrigido.index.min() == pd.Timestamp("2010-01-01 00:00")
    assert corrigido["temperature_2m"].iloc[0] == 0  # os valores não mudam, só o índice


def test_carga_ja_em_horario_local_nao_e_alterada(tmp_path):
    idx = pd.date_range("2010-01-01 00:00", periods=24, freq="h")
    df = pd.DataFrame({"temperature_2m": range(24)}, index=idx)
    corrigido, audit = ensure_local_time(df, _meta(tmp_path))
    assert audit["legacy_utc_detected"] is False
    assert corrigido.index.equals(df.index)


def test_metadado_de_horario_local_dispensa_correcao(tmp_path):
    idx = pd.date_range("2010-01-01 03:00", periods=24, freq="h")
    df = pd.DataFrame({"temperature_2m": range(24)}, index=idx)
    _, audit = ensure_local_time(df, _meta(tmp_path, time_reference="horario_local_padrao_utc-03:00"))
    assert audit["legacy_utc_detected"] is False


def test_lotes_incrementais_sao_unidos_sem_duplicatas(tmp_path):
    a = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3, freq="h"), "temperature_2m": [1, 2, 3]})
    b = pd.DataFrame({"date": pd.date_range("2025-01-01 02:00", periods=2, freq="h"), "temperature_2m": [30, 40]})
    a.to_csv(tmp_path / "bauru_weather_hourly_incremental_20250101_20250101.csv", index=False)
    b.to_csv(tmp_path / "bauru_weather_hourly_incremental_20250102_20250102.csv", index=False)
    df = load_incremental_raws(str(tmp_path))
    assert len(df) == 4
    assert df.loc["2025-01-01 02:00", "temperature_2m"] == 30  # vale o lote mais recente


def test_eventos_de_estiagem_a_partir_do_contador():
    contador = [0, 1, 2, 3, 0, 0, 1, 0, 1, 2]
    df = pd.DataFrame({"consecutive_dry_days": contador}, index=pd.date_range("2020-01-01", periods=10, freq="D"))
    eventos = extract_dry_spell_events(df)
    assert eventos["duration_days"].tolist() == [3, 1, 2]
    assert eventos["start"].iloc[0] == pd.Timestamp("2020-01-02")
    assert eventos["end"].iloc[-1] == pd.Timestamp("2020-01-10")  # evento ainda em curso no fim da série
