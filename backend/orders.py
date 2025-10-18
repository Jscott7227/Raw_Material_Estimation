"""Helper utilities for loading planned outbound orders."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

ORDERS_CSV = Path(__file__).parent / "data" / "orders.csv"


@dataclass(frozen=True)
class OrderRecord:
    order_id: str
    material: str
    requested_date: datetime
    required_tons: float
    status: str


def _parse_row(row: dict) -> OrderRecord:
    requested = datetime.fromisoformat(row["requested_date"]).replace(tzinfo=timezone.utc)
    required_tons = float(row.get("required_weight_tons", 0.0))
    return OrderRecord(
        order_id=row["order_id"],
        material=row["material"].strip(),
        requested_date=requested,
        required_tons=required_tons,
        status=row.get("status", "").strip().lower() or "pending",
    )


@lru_cache(maxsize=1)
def _load_orders_cached(timestamp: float) -> List[OrderRecord]:
    _ = timestamp  # cache key ensures refresh when file timestamp changes
    if not ORDERS_CSV.exists():
        return []
    with ORDERS_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [_parse_row(row) for row in reader]


def load_orders() -> List[OrderRecord]:
    """Return all planned orders (cached unless file timestamp changes)."""
    try:
        mtime = ORDERS_CSV.stat().st_mtime
    except FileNotFoundError:
        return []
    return _load_orders_cached(mtime)


__all__ = ["OrderRecord", "load_orders"]
