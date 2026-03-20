"""Utility helpers for normalization and parsing."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, datetime, time, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from scada_faults.config import EXPECTED_YEAR_RANGE, TIME_MISSING_TOKENS, TIME_SPECIAL_TOKENS


def normalize_string(value: object, *, lowercase: bool = False) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    lowered = text.lower()
    if lowered in TIME_MISSING_TOKENS:
        return None
    return lowered if lowercase else text


def canonical_lower(value: object) -> str:
    text = normalize_string(value, lowercase=True)
    return text or ""


def parse_mixed_date(value: object) -> pd.Timestamp:
    text = normalize_string(value)
    if not text:
        return pd.NaT
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?", text):
        return pd.to_datetime(text[:10], format="%Y-%m-%d", errors="coerce")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def parse_clock_time(value: object) -> tuple[pd.Timestamp | pd.NaT, str]:
    text = canonical_lower(value)
    if not text:
        return pd.NaT, "missing"
    if text in TIME_SPECIAL_TOKENS:
        return pd.NaT, text
    parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT, "unparsed"
    return parsed, "clock"


def combine_date_and_time(event_date: pd.Timestamp, event_time: pd.Timestamp | pd.NaT) -> pd.Timestamp:
    if pd.isna(event_date) or pd.isna(event_time):
        return pd.NaT
    return pd.Timestamp.combine(event_date.date(), event_time.time())


def combine_reclose_datetime(
    trip_datetime: pd.Timestamp,
    event_date: pd.Timestamp,
    reclose_time: pd.Timestamp | pd.NaT,
) -> pd.Timestamp:
    if pd.isna(event_date) or pd.isna(reclose_time):
        return pd.NaT
    reclose_dt = pd.Timestamp.combine(event_date.date(), reclose_time.time())
    if not pd.isna(trip_datetime) and reclose_dt < trip_datetime:
        reclose_dt = reclose_dt + timedelta(days=1)
    return reclose_dt


def parse_float(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, float):
        return value
    text = normalize_string(value)
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def normalize_weather(value: object) -> str:
    text = canonical_lower(value)
    if not text:
        return "unknown"
    if "rain" in text or "storm" in text or "thunder" in text:
        return "rainy/stormy"
    if "cloud" in text:
        return "cloudy"
    if "sun" in text or "clear" in text or "normal" in text:
        return "clear"
    return text


def normalize_reporter(value: object) -> str:
    text = canonical_lower(value)
    if not text:
        return "unknown"
    text = text.replace(" / ", "/")
    text = text.replace("scada /", "scada/")
    text = text.replace("scada/", "")
    text = text.replace("scada", "scada")
    text = text.strip("/")
    if not text:
        return "scada"
    return text


def season_from_month(month: int | float | None) -> str:
    if month is None or pd.isna(month):
        return "unknown"
    month_int = int(month)
    if month_int in {12, 1, 2}:
        return "summer"
    if month_int in {3, 4, 5}:
        return "autumn"
    if month_int in {6, 7, 8}:
        return "winter"
    return "spring"


def mode_or_unknown(values: Iterable[object]) -> str:
    normalized = [normalize_string(v) for v in values if normalize_string(v)]
    if not normalized:
        return "unknown"
    return Counter(normalized).most_common(1)[0][0]


def join_unique(values: Iterable[object]) -> str:
    seen: list[str] = []
    for value in values:
        text = normalize_string(value)
        if text and text not in seen:
            seen.append(text)
    return " || ".join(seen)


def normalize_operational_label(value: object) -> str:
    text = canonical_lower(value)
    return text or "unknown"


def infer_anomaly_reason(fault_no: str, event_date: pd.Timestamp) -> str:
    if fault_no in {"117/23", "118/23"}:
        return "fault_id_date_anomaly"
    if pd.isna(event_date):
        return "unparsed_date"
    min_year, max_year = EXPECTED_YEAR_RANGE
    if event_date.year < min_year or event_date.year > max_year:
        return "out_of_expected_range"
    return ""
