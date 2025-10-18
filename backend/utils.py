"""Utility helpers shared across the backend services."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

# Accepted truck delivery statuses. All values are stored in lowercase form.
_DELIVERY_STATUSES: set[str] = {"completed", "upcoming"}


def normalize_delivery_status(value: str | None, *, default: str = "upcoming") -> str:
    """Return a canonical delivery status value.

    Any value outside the supported set falls back to the supplied *default*
    (which is validated to ensure it is part of the allowed statuses).
    """
    if default not in _DELIVERY_STATUSES:
        raise ValueError(f"Invalid default status '{default}'")

    if value is None:
        return default

    normalized = value.strip().lower()
    return normalized if normalized in _DELIVERY_STATUSES else default


def parse_delivery_timestamp(raw: str | datetime | None) -> datetime:
    """Parse a delivery timestamp into an aware UTC datetime."""
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)

    text = (raw or "").strip()
    if not text:
        return datetime.now(timezone.utc)

    parsers: Iterable[str] = (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    )

    # First try Python's ISO parser which covers most cases, then iterate fallbacks.
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    except ValueError:
        pass

    for fmt in parsers:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return datetime.now(timezone.utc)


__all__ = ["normalize_delivery_status", "parse_delivery_timestamp"]
