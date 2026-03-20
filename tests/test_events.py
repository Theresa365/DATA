from __future__ import annotations

import pandas as pd

from scada_faults.events import aggregate_events, worst_case_operational_label


def test_worst_case_operational_label_uses_planned_precedence() -> None:
    label = worst_case_operational_label(["transient", "sustained", "permanent"])
    assert label == "permanent"


def test_aggregate_events_collapses_multirow_incident() -> None:
    rows = pd.DataFrame(
        [
            {
                "source_name": "distribution",
                "fault_no": "062/24",
                "event_date_key": "2024-03-30",
                "trip_time_key": "17:05",
                "event_date": pd.Timestamp("2024-03-30"),
                "trip_datetime": pd.Timestamp("2024-03-30 17:05"),
                "comments_raw": "OC & E/F trip",
                "apparatus_tripped": "Kanye T1 LV CB 1HO",
                "system_type_clean": "Distribution",
                "substation_area_clean": "South",
                "weather_clean": "clear",
                "reported_by_clean": "mosarwe",
                "voltage_level_kv_num": 11.0,
                "voltage_level_kv": "11",
                "downtime_hours": 4.28,
                "reclose_delay_hours": 4.28,
                "time_reclosed_status": "clock",
                "fault_type_clean": "permanent",
                "is_chrono_anomaly": False,
                "anomaly_reason": "",
                "location_town": "Kanye",
            },
            {
                "source_name": "distribution",
                "fault_no": "062/24",
                "event_date_key": "2024-03-30",
                "trip_time_key": "17:05",
                "event_date": pd.Timestamp("2024-03-30"),
                "trip_datetime": pd.Timestamp("2024-03-30 17:05"),
                "comments_raw": "DID NOT TRIP",
                "apparatus_tripped": "Kanye T1 HV CB 180",
                "system_type_clean": "Distribution",
                "substation_area_clean": "South",
                "weather_clean": "clear",
                "reported_by_clean": "mosarwe",
                "voltage_level_kv_num": 11.0,
                "voltage_level_kv": "11",
                "downtime_hours": 0.0,
                "reclose_delay_hours": pd.NA,
                "time_reclosed_status": "missing",
                "fault_type_clean": "transient",
                "is_chrono_anomaly": False,
                "anomaly_reason": "",
                "location_town": "Kanye",
            },
        ]
    )

    events = aggregate_events(rows)
    assert len(events) == 1
    record = events.iloc[0]
    assert record["row_count"] == 2
    assert record["unique_apparatus_count"] == 2
    assert record["stage1_operational_label"] == "permanent"
    assert record["stage1_binary_label"] == "Permanent"
    assert "Kanye T1 LV CB 1HO" in record["apparatus_concat"]
    assert "DID NOT TRIP" in record["comments_concat"]
