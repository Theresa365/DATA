from __future__ import annotations

import pandas as pd

from scada_faults.utils import infer_anomaly_reason, parse_clock_time, parse_mixed_date


def test_parse_mixed_date_handles_both_supported_formats() -> None:
    assert parse_mixed_date("18/04/2022") == pd.Timestamp("2022-04-18")
    assert parse_mixed_date("2024-03-16") == pd.Timestamp("2024-03-16")
    assert parse_mixed_date("2024-03-16 00:00:00") == pd.Timestamp("2024-03-16")


def test_parse_clock_time_handles_clock_and_special_tokens() -> None:
    parsed, status = parse_clock_time("08:15")
    assert status == "clock"
    assert parsed.strftime("%H:%M") == "08:15"

    parsed_arc, status_arc = parse_clock_time("ARC")
    assert pd.isna(parsed_arc)
    assert status_arc == "arc"

    parsed_nil, status_nil = parse_clock_time("NIL")
    assert pd.isna(parsed_nil)
    assert status_nil == "missing"


def test_infer_anomaly_reason_flags_known_fault_ids() -> None:
    assert infer_anomaly_reason("117/23", pd.Timestamp("2022-04-18")) == "fault_id_date_anomaly"
    assert infer_anomaly_reason("999/24", pd.Timestamp("2024-03-16")) == ""
