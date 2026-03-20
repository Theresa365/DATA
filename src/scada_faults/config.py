"""Project configuration constants."""

from __future__ import annotations

RAW_COLUMN_MAP = {
    "Date": "date_raw",
    "Trip Time": "trip_time_raw",
    "Fault No": "fault_no",
    "Location (Town)": "location_town",
    "Apparatus Tripped": "apparatus_tripped",
    "Voltage Level(KV)": "voltage_level_kv",
    "System Type": "system_type",
    "Substation Area": "substation_area",
    "Weather": "weather_raw",
    "Time Reclosed": "time_reclosed_raw",
    "Downtime (Hours)": "downtime_hours_raw",
    "Reported By": "reported_by_raw",
    "Comments": "comments_raw",
    "Month": "month_raw",
    "Severity": "severity_raw",
    "Fault type": "fault_type_raw",
}

ROW_ID_COLUMNS = list(RAW_COLUMN_MAP.values())

DATE_ANOMALY_FAULT_IDS = {"117/23", "118/23"}
EXPECTED_YEAR_RANGE = (2023, 2025)

OPERATIONAL_WORST_CASE_ORDER = {
    "false": 0,
    "transient": 1,
    "sustained": 2,
    "permanent": 3,
}

STAGE2_LABELS = [
    "ground-related",
    "phase-to-phase",
    "three-phase",
    "transformer/internal",
    "operational-other",
    "unknown/unclassifiable",
]

TIME_MISSING_TOKENS = {"", "na", "n/a", "nan", "nil", "none", "null"}
TIME_SPECIAL_TOKENS = {"arc"}
DEFAULT_RANDOM_STATE = 42

TEXT_JOIN_TOKEN = " || "
