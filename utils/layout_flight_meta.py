"""Helpers for flight records stored in layout JSON (service dates, etc.)."""
from __future__ import annotations

from typing import Any


def ensure_flight_service_dates(layout: dict[str, Any], default_iso: str) -> None:
    """Set ``flight[\"serviceDate\"]`` (YYYY-MM-DD) when missing; keep existing values."""
    if not isinstance(layout, dict):
        return
    flights = layout.get("flights")
    if not isinstance(flights, list):
        return
    d = (default_iso or "").strip() or "2026-03-31"
    for f in flights:
        if not isinstance(f, dict):
            continue
        raw = f.get("serviceDate")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            f["serviceDate"] = d
