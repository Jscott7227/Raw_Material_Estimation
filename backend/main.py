import asyncio
import random
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Deque, Generator, List, Optional, Tuple
from uuid import uuid4
from img_model import calc_weight
import cv2
import numpy as np

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import BytesIO
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import SessionLocal, init_db
from load_shipments import (
    DEFAULT_JSON as SHIPMENTS_DEFAULT_JSON,
    canonical_material_name,
    load_json_records,
)
from models import Material, TruckDelivery, MaterialInventoryHistory
from schemas import (
    BillOfLading,
    BillOfLadingSimulationConfig,
    DemoStatus,
    DemoSeedRequest,
    MaterialAlert,
    MaterialRead,
    MaterialRecommendation,
    MaterialCreate,
    MaterialUpdate,
    MaterialAdjustRequest,
    DeliveryCreateRequest,
    SensorEvent,
    TruckDeliveryRead,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # Create tables and preload materials
    try:
        yield
    finally:
        await _stop_demo()

app = FastAPI(
    title="Raw Materials API",
    description="API for Portobello America's raw materials dashboard.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # open for hackathon simplicity
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


CARRIER_CHOICES = [
    "L&H Express",
    "Swift Logistics",
    "Portobello Fleet",
    "Clay Haulers Co.",
    "Blue Ridge Transit",
]

BIN_CAPACITY_TONS = 1200.0
REORDER_THRESHOLD_RATIO = 0.5
WARNING_THRESHOLD_RATIO = 0.65
ORDER_LEAD_TIME_DAYS = 4
RESTOCK_TARGET_RATIO = 0.95

DEMO_STATE: dict = {
    "task": None,
    "start_time": None,
    "duration": None,
    "delivery_queue": deque(),
}

# Demo loops use these constants; replace with real recipe or schedule once production ready.
DEMO_CONSUMPTION_TONS = 2.5  # TODO: replace with actual batch size (tons) derived from live recipe data
DEMO_CONSUMPTION_INTERVAL = 1.5  # seconds between simulated consumption batches
DEMO_DELIVERY_INTERVAL = 3.0  # seconds between simulated inbound deliveries
DEMO_DELIVERY_CONFIG = BillOfLadingSimulationConfig(
    min_trucks_per_day=60,  # TODO: replace with real truck arrival rates
    max_trucks_per_day=120,
    min_tons=18.0,  # TODO: replace with actual expected load ranges
    max_tons=32.0,
)


def _material_metrics(material: Material) -> dict:
    fill_ratio = material.weight / BIN_CAPACITY_TONS if BIN_CAPACITY_TONS else 0.0
    bins_filled = material.weight / BIN_CAPACITY_TONS if BIN_CAPACITY_TONS else 0.0
    return {
        "id": material.id,
        "type": material.type,
        "weight": material.weight,
        "humidity": material.humidity,
        "density": material.density,
        "bin_capacity": BIN_CAPACITY_TONS,
        "fill_ratio": max(fill_ratio, 0.0),
        "bins_filled": max(bins_filled, 0.0),
        "needs_reorder": material.weight <= BIN_CAPACITY_TONS * REORDER_THRESHOLD_RATIO,
    }


def _record_history(db: Session, materials: List[Material]) -> None:
    """Persist an inventory snapshot for each material in *materials*."""
    timestamp = datetime.utcnow()
    for material in materials:
        db.add(
            MaterialInventoryHistory(
                material_id=material.id,
                weight=material.weight,
                recorded_at=timestamp,
            )
        )


def _material_alerts(material: Material) -> List[MaterialAlert]:
    """Return any alert records associated with the current material level."""
    metrics = _material_metrics(material)
    alerts: List[MaterialAlert] = []
    ratio = metrics["fill_ratio"]

    if ratio <= REORDER_THRESHOLD_RATIO:
        alerts.append(
                MaterialAlert(
                    material_id=material.id,
                    material_type=material.type,
                    weight=material.weight,
                    fill_ratio=ratio,
                    bins_filled=metrics["bins_filled"],
                    alert_level="critical",
                    message=(
                        f"{material.type} bin below 50% capacity "
                        f"({material.weight:.1f} tons remaining; {(ratio * 100):.0f}% full). "
                        f"Lead time is {ORDER_LEAD_TIME_DAYS} days—place an order now."
                    ),
                )
            )

    return alerts


def _demo_running() -> bool:
    task = DEMO_STATE["task"]
    return task is not None and not task.done()


def _demo_seconds_remaining() -> Optional[float]:
    if not _demo_running():
        return None
    start = DEMO_STATE.get("start_time")
    duration = DEMO_STATE.get("duration")
    if not start or not duration:
        return None
    elapsed = (datetime.utcnow() - start).total_seconds()
    return max(0.0, duration - elapsed)


def _generate_recommendations(db: Session, days: int = 7) -> List[MaterialRecommendation]:
    """Forecast depletion for each material using trailing history and return order advice."""
    lookback = datetime.utcnow() - timedelta(days=days)
    recommendations: List[MaterialRecommendation] = []

    materials = db.query(Material).all()
    for material in materials:
        history = (
            db.query(MaterialInventoryHistory)
            .filter(
                MaterialInventoryHistory.material_id == material.id,
                MaterialInventoryHistory.recorded_at >= lookback,
            )
            .order_by(MaterialInventoryHistory.recorded_at.asc())
            .all()
        )

        if not history:
            history = (
                db.query(MaterialInventoryHistory)
                .filter(MaterialInventoryHistory.material_id == material.id)
                .order_by(MaterialInventoryHistory.recorded_at.desc())
                .limit(10)
                .all()
            )
            history = list(reversed(history))

        if len(history) < 2:
            recommendations.append(
                MaterialRecommendation(
                    material_id=material.id,
                    material_type=material.type,
                    current_weight=material.weight,
                    average_daily_consumption=0.0,
                    projected_weight_seven_days=material.weight,
                    days_until_reorder=None,
                    recommended_order_date=None,
                    recommended_order_tons=None,
                    rationale="Insufficient history to estimate consumption",
                )
            )
            continue

        total_consumption = 0.0
        for prev, curr in zip(history, history[1:]):
            delta = prev.weight - curr.weight
            if delta > 0:
                total_consumption += delta

        span_seconds = (history[-1].recorded_at - history[0].recorded_at).total_seconds()
        span_days = max(span_seconds / 86400, 1.0)
        avg_daily = total_consumption / span_days if total_consumption > 0 else 0.0

        projected_weight = material.weight - avg_daily * days
        reorder_threshold = BIN_CAPACITY_TONS * REORDER_THRESHOLD_RATIO
        lead_time = ORDER_LEAD_TIME_DAYS
        days_until_reorder: Optional[float] = None
        order_offset: Optional[float] = None

        if avg_daily > 0:
            days_until_reorder = max(
                0.0, (material.weight - reorder_threshold) / avg_daily
            )
            order_offset = max(days_until_reorder - lead_time, 0.0)

        recommended_tons = None
        recommended_date = None
        rationale = "Inventory stable; no reorder expected within outlook"

        if avg_daily <= 0.01:
            rationale = "No consumption detected; monitoring only"
        else:
            target_weight = BIN_CAPACITY_TONS * RESTOCK_TARGET_RATIO
            lead_consumption = avg_daily * lead_time
            projected_at_arrival = material.weight - lead_consumption
            refill_amount = max(0.0, target_weight - projected_at_arrival)

            if refill_amount > 0:
                recommended_tons = round(refill_amount, 2)
                max_capacity_delta = max(0.0, BIN_CAPACITY_TONS - material.weight)
                if max_capacity_delta > 0:
                    recommended_tons = min(recommended_tons, round(max_capacity_delta, 2))

            if days_until_reorder is not None:
                if days_until_reorder <= lead_time:
                    recommended_date = datetime.utcnow()
                    rationale = (
                        f"Reorder threshold hit in {days_until_reorder:.1f} days. "
                        f"With a {lead_time}-day lead time, place an order immediately."
                    )
                elif order_offset is not None and order_offset <= days:
                    recommended_date = datetime.utcnow() + timedelta(days=order_offset)
                    rationale = (
                        f"Reorder threshold expected in {days_until_reorder:.1f} days. "
                        f"Order in {order_offset:.1f} days ({recommended_date.date()}) to account for "
                        f"the {lead_time}-day lead time."
                    )
                elif projected_weight <= reorder_threshold:
                    recommended_date = datetime.utcnow() + timedelta(days=days)
                    rationale = (
                        "Inventory projected to fall below threshold beyond current outlook; "
                        "begin planning for a replenishment."
                    )
                else:
                    recommended_date = None
                    recommended_tons = None
                    rationale = (
                        "Inventory remains above reorder threshold for the selected horizon."
                    )

        recommendations.append(
            MaterialRecommendation(
                material_id=material.id,
                material_type=material.type,
                current_weight=material.weight,
                average_daily_consumption=round(avg_daily, 2),
                projected_weight_seven_days=round(projected_weight, 2),
                days_until_reorder=round(days_until_reorder, 2) if days_until_reorder is not None else None,
                recommended_order_date=recommended_date,
                recommended_order_tons=round(recommended_tons, 2) if recommended_tons is not None else None,
                rationale=rationale,
            )
        )

    return recommendations


def _generate_recipe(materials: List[Material], min_percent: float) -> List[float]:
    if not materials:
        return []
    # TODO: replace heuristic allocation with actual recipe percentages per material.
    # For production, read formula ratios from your MES/ERP and remove the random draw below.

    # Cap the minimum to avoid exceeding 100% while using random fallback
    min_cap = min(min_percent, 100 / len(materials))

    draws = [random.random() for _ in materials]
    total = sum(draws) or 1
    base = [(value / total) * 100 for value in draws]

    clipped = [max(min_cap, percent) for percent in base]
    scale = 100 / sum(clipped)
    scaled = [percent * scale for percent in clipped]

    # Fix rounding drift on the last element
    rounded = [round(percent, 2) for percent in scaled]
    drift = 100 - sum(rounded)
    rounded[-1] = round(rounded[-1] + drift, 2)
    return rounded


def _apply_recipe_batch(
    db: Session,
    total_tons: float,
    min_percent: float,
) -> List[SensorEvent]:
    materials = db.query(Material).all()
    if not materials or total_tons <= 0:
        return []

    percentages = _generate_recipe(materials, min_percent)
    events: List[SensorEvent] = []
    changed_materials: List[Material] = []

    for material, percent in zip(materials, percentages):
        draw = round(total_tons * (percent / 100), 2)
        if draw <= 0:
            continue

        delta = -draw
        material.apply_weight_change(delta)
        changed_materials.append(material)
        events.append(
            SensorEvent(
                material_id=material.id,
                material_type=material.type,
                delta=delta,
                remaining_weight=material.weight,
                timestamp=datetime.utcnow(),
            )
        )

    db.commit()
    return events


def _generate_bol_payload(
    db: Session, config: BillOfLadingSimulationConfig
) -> Optional[BillOfLading]:
    materials = db.query(Material).all()
    if not materials:
        return None

    material: Optional[Material] = None
    target_ratio: Optional[float] = None

    queue: Deque[Tuple[int, float]] = DEMO_STATE.get("delivery_queue")  # type: ignore
    if queue:
        while queue:
            forced_id, forced_ratio = queue.popleft()
            candidate = next((item for item in materials if item.id == forced_id), None)
            if candidate:
                material = candidate
                target_ratio = forced_ratio
                break

    if material is None or target_ratio is None:
        selection = _select_demo_delivery_material(materials)
        if not selection:
            return None
        material, target_ratio = selection
    deficit_tons = max(BIN_CAPACITY_TONS * target_ratio - material.weight, 0.0)
    if deficit_tons <= 0:
        return None

    tons = min(max(deficit_tons, config.min_tons), config.max_tons)
    tons = round(tons, 2)
    net_lb = round(tons * 2000, 2)
    tare_lb = round(random.uniform(25000, 32000), 2)
    gross_lb = round(net_lb + tare_lb, 2)

    delivery_number = f"SIM-{int(datetime.utcnow().timestamp())}-{random.randint(1000, 9999)}"
    carrier = random.choice(CARRIER_CHOICES)

    return BillOfLading(
        delivery_number=delivery_number,
        po_number=f"PO-{uuid4().hex[:6].upper()}",
        customer="Portobello Americas Inc.",
        product=material.type.replace(" ", "_").upper(),
        net_weight_lb=net_lb,
        gross_weight_lb=gross_lb,
        tare_weight_lb=tare_lb,
        carrier=carrier,
        delivery_date=datetime.utcnow(),
        material_code=material.type,
    )


async def _demo_runner(duration_seconds: int) -> None:
    """Drive the scripted demo loop (consumption + deliveries) for *duration_seconds*."""
    loop = asyncio.get_running_loop()
    end_time = loop.time() + duration_seconds

    next_consumption = loop.time()
    next_delivery = loop.time()

    try:
        while loop.time() < end_time:
            now = loop.time()

            if now >= next_consumption:
                session = SessionLocal()
                try:
                    _apply_recipe_batch(
                        session,
                        total_tons=DEMO_CONSUMPTION_TONS,
                        min_percent=5.0,
                    )
                finally:
                    session.close()
                next_consumption += DEMO_CONSUMPTION_INTERVAL

            if now >= next_delivery:
                session = SessionLocal()
                try:
                    payload = _generate_bol_payload(session, DEMO_DELIVERY_CONFIG)
                    if payload:
                        _process_bill_of_lading(session, payload)
                finally:
                    session.close()
                next_delivery += DEMO_DELIVERY_INTERVAL

            await asyncio.sleep(0.25)
    finally:
        DEMO_STATE.update({"task": None, "start_time": None, "duration": None})


async def _stop_demo() -> None:
    task: Optional[asyncio.Task] = DEMO_STATE["task"]
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    DEMO_STATE.update({"task": None, "start_time": None, "duration": None})
    DEMO_STATE["delivery_queue"] = deque()


@app.get("/api/health")
def health_check():
    return {"status": "running"}

@app.get("/api/materials", response_model=List[MaterialRead])
def get_materials(db: Session = Depends(get_db)) -> List[MaterialRead]:
    materials = db.query(Material).all()
    return [
        MaterialRead.model_validate(_material_metrics(material))
        for material in materials
    ]


@app.post("/api/materials", response_model=MaterialRead, status_code=201)
def create_material(
    payload: MaterialCreate,
    db: Session = Depends(get_db),
) -> MaterialRead:
    """Provision a new raw material (hook this to master-data ingestion in production)."""
    existing = (
        db.query(Material)
        .filter(func.lower(Material.type) == payload.type.lower())
        .one_or_none()
    )
    if existing:
        raise HTTPException(status_code=409, detail="Material already exists")

    material = Material(
        type=payload.type,
        weight=payload.weight,
        humidity=payload.humidity,
        density=payload.density,
    )
    db.add(material)
    db.flush()
    _record_history(db, [material])
    db.commit()
    db.refresh(material)
    return MaterialRead.model_validate(_material_metrics(material))


@app.put("/api/materials/{material_id}", response_model=MaterialRead)
def update_material(
    material_id: int,
    payload: MaterialUpdate,
    db: Session = Depends(get_db),
) -> MaterialRead:
    """Update material metadata or on-hand weight from your live system."""
    material = db.get(Material, material_id)
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")

    if payload.type and payload.type.lower() != material.type.lower():
        conflict = (
            db.query(Material)
            .filter(func.lower(Material.type) == payload.type.lower(), Material.id != material_id)
            .one_or_none()
        )
        if conflict:
            raise HTTPException(status_code=409, detail="Material name already in use")
        material.type = payload.type

    weight_changed = False
    if payload.weight is not None:
        material.weight = payload.weight
        weight_changed = True
    if payload.humidity is not None:
        material.humidity = payload.humidity
    if payload.density is not None:
        material.density = payload.density

    if weight_changed:
        _record_history(db, [material])

    db.commit()
    db.refresh(material)
    return MaterialRead.model_validate(_material_metrics(material))


@app.post("/api/materials/{material_id}/adjust", response_model=MaterialRead)
def adjust_material(
    material_id: int,
    payload: MaterialAdjustRequest,
    db: Session = Depends(get_db),
) -> MaterialRead:
    """Apply ad-hoc adjustments (e.g., manual counts, QA scrap)."""
    material = db.get(Material, material_id)
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")

    material.apply_weight_change(payload.delta)
    _record_history(db, [material])
    db.commit()
    db.refresh(material)
    return MaterialRead.model_validate(_material_metrics(material))


@app.get("/api/alerts", response_model=List[MaterialAlert])
def get_alerts(db: Session = Depends(get_db)) -> List[MaterialAlert]:
    materials = db.query(Material).all()
    alerts: List[MaterialAlert] = []
    for material in materials:
        alerts.extend(_material_alerts(material))
    return alerts


@app.get("/api/recommendations", response_model=List[MaterialRecommendation])
def get_recommendations(days: int = 7, db: Session = Depends(get_db)) -> List[MaterialRecommendation]:
    days = max(1, min(days, 30))
    return _generate_recommendations(db, days=days)


def _distribute_consumption(total: float, periods: int) -> List[float]:
    if total <= 0 or periods <= 0:
        return [0.0] * max(periods, 0)
    draws = [random.uniform(0.6, 1.4) for _ in range(periods)]
    scale = total / (sum(draws) or 1.0)
    values = [round(draw * scale, 2) for draw in draws]
    drift = round(total - sum(values), 2)
    values[-1] = round(values[-1] + drift, 2)
    return values


def _shape_demo_levels(materials: List[Material]) -> List[Material]:
    if not materials:
        return []
    ordered = sorted(materials, key=lambda material: material.id)
    profiles = [0.94, 0.82, 0.68, 0.48, 0.42, 0.59, 0.75]
    for index, material in enumerate(ordered):
        ratio = profiles[index] if index < len(profiles) else profiles[-1]
        target_weight = round(BIN_CAPACITY_TONS * ratio, 2)
        material.weight = max(0.0, min(target_weight, BIN_CAPACITY_TONS))
    return ordered


def _material_fill_ratio(material: Material) -> float:
    if BIN_CAPACITY_TONS <= 0:
        return 0.0
    return max(0.0, material.weight / BIN_CAPACITY_TONS)


def _select_demo_delivery_material(materials: List[Material]) -> Optional[Tuple[Material, float]]:
    if not materials:
        return None

    lows = [
        material
        for material in materials
        if _material_fill_ratio(material) <= REORDER_THRESHOLD_RATIO
    ]
    if lows:
        material = min(lows, key=_material_fill_ratio)
        target = random.uniform(0.9, 0.96)
        return material, target

    mediums = [
        material
        for material in materials
        if REORDER_THRESHOLD_RATIO < _material_fill_ratio(material) <= WARNING_THRESHOLD_RATIO
    ]
    if mediums and random.random() < 0.7:
        material = min(mediums, key=_material_fill_ratio)
        target = random.uniform(0.83, 0.9)
        return material, target

    highs = [
        material
        for material in materials
        if _material_fill_ratio(material) > WARNING_THRESHOLD_RATIO
    ]
    if highs and random.random() < 0.2:
        material = random.choice(highs)
        current_ratio = _material_fill_ratio(material)
        baseline = random.uniform(0.76, 0.85)
        target = max(baseline, current_ratio + 0.02)
        target = min(target, 0.95)
        return material, target

    return None


def _seed_demo_history(
    db: Session,
    materials: List[Material],
    *,
    days: int = 7,
    replace: bool = True,
) -> None:
    if not materials:
        return

    base_time = datetime.utcnow() - timedelta(days=days)

    for material in materials:
        if replace:
            db.query(MaterialInventoryHistory).filter(
                MaterialInventoryHistory.material_id == material.id
            ).delete()
        current_weight = float(material.weight or 0.0)
        extra = random.uniform(80, 220)
        start_weight = min(
            BIN_CAPACITY_TONS * 0.98,
            max(current_weight + 40.0, current_weight + extra),
        )
        total_consumption = max(start_weight - current_weight, 0.0)
        if total_consumption <= 1e-2:
            total_consumption = max(60.0, current_weight * 0.25)
            start_weight = min(
                BIN_CAPACITY_TONS * 0.98, current_weight + total_consumption
            )

        consumption_series = _distribute_consumption(total_consumption, days)
        timestamp = base_time
        running_weight = start_weight

        for draw in consumption_series:
            db.add(
                MaterialInventoryHistory(
                    material_id=material.id,
                    weight=round(max(running_weight, 0.0), 2),
                    recorded_at=timestamp,
                )
            )
            running_weight = max(running_weight - draw, 0.0)
            timestamp += timedelta(days=1)

        db.add(
            MaterialInventoryHistory(
                material_id=material.id,
                weight=round(current_weight, 2),
                recorded_at=datetime.utcnow(),
            )
        )


def _add_future_delivery(
    db: Session,
    materials: List[Material],
    *,
    status: str,
    material: Optional[Material] = None,
    target_ratio: Optional[float] = None,
) -> None:
    if not materials and material is None:
        return
    if material is None:
        material = random.choice(materials)
    horizon_hours = (4, 12) if status == "pending" else (12, 36)
    delivery_time = (datetime.utcnow() + timedelta(hours=random.randint(*horizon_hours))).strftime(
        "%Y-%m-%d %H:%M"
    )
    if target_ratio is None:
        target_ratio = 0.92 if status == "pending" else 0.78
    target_ratio = max(0.55, min(target_ratio, 0.97))
    target_weight = BIN_CAPACITY_TONS * target_ratio
    deficit_tons = max(target_weight - material.weight, 0.0)
    if deficit_tons <= 0:
        return
    incoming_tons = min(
        max(deficit_tons, DEMO_DELIVERY_CONFIG.min_tons),
        DEMO_DELIVERY_CONFIG.max_tons,
    )
    incoming_weight_lb = round(incoming_tons * 2000, 2)
    delivery = TruckDelivery(
        material_id=material.id,
        delivery_num=f"DEMO-{uuid4().hex[:6].upper()}-{status[:3].upper()}",
        incoming_weight=incoming_weight_lb,
        delivery_time=delivery_time,
        status=status,
    )
    db.add(delivery)


def _ensure_demo_deliveries(db: Session, materials: List[Material]) -> None:
    try:
        records = list(load_json_records(SHIPMENTS_DEFAULT_JSON))
    except FileNotFoundError:
        records = []
    except Exception:
        records = []

    existing_numbers = {
        delivery.delivery_num for delivery in db.query(TruckDelivery).all()
    }
    material_index = {
        material.type.strip().lower(): material for material in materials
    }

    for entry in records:
        delivery_number = entry.get("deliveryNumber")
        if not delivery_number or delivery_number in existing_numbers:
            continue
        canonical = canonical_material_name(entry.get("material", "")).strip().lower()
        material = material_index.get(canonical)
        if not material:
            continue
        status = entry.get("status", "pending").lower()
        if status not in {"pending", "completed", "upcoming"}:
            status = "pending"
        db.add(
            TruckDelivery(
                material_id=material.id,
                delivery_num=delivery_number,
                incoming_weight=float(entry.get("incomingWeight", 0.0)),
                delivery_time=entry.get(
                    "deliveryDateTime",
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                ),
                status=status,
            )
        )
        existing_numbers.add(delivery_number)

    db.flush()

    if not db.query(TruckDelivery).filter(TruckDelivery.status == "upcoming").first():
        _add_future_delivery(db, materials, status="upcoming")
    if not db.query(TruckDelivery).filter(TruckDelivery.status == "pending").first():
        _add_future_delivery(db, materials, status="pending")

    reorder_threshold = BIN_CAPACITY_TONS * REORDER_THRESHOLD_RATIO
    for material in materials:
        if material.weight <= reorder_threshold:
            existing_pending = (
                db.query(TruckDelivery)
                .filter(
                    TruckDelivery.material_id == material.id,
                    TruckDelivery.status == "pending",
                )
                .first()
            )
            if not existing_pending:
                _add_future_delivery(
                    db,
                    materials,
                    status="pending",
                    material=material,
                    target_ratio=0.92,
                )
            existing_upcoming = (
                db.query(TruckDelivery)
                .filter(
                    TruckDelivery.material_id == material.id,
                    TruckDelivery.status == "upcoming",
                )
                .first()
            )
            if not existing_upcoming:
                _add_future_delivery(
                    db,
                    materials,
                    status="upcoming",
                    material=material,
                    target_ratio=0.8,
                )
        elif material.weight <= reorder_threshold * 1.1:
            existing_upcoming = (
                db.query(TruckDelivery)
                .filter(
                    TruckDelivery.material_id == material.id,
                    TruckDelivery.status == "upcoming",
                )
                .first()
            )
            if not existing_upcoming:
                _add_future_delivery(
                    db,
                    materials,
                    status="upcoming",
                    material=material,
                    target_ratio=0.85,
                )

    focus_materials = {
        "sms clay": {
            "pending": {"count": 3, "target_ratio": 0.96},
            "upcoming": {"count": 2, "target_ratio": 0.9},
        },
        "feldspar": {
            "pending": {"count": 3, "target_ratio": 0.95},
            "upcoming": {"count": 2, "target_ratio": 0.88},
        },
    }
    for name, targets in focus_materials.items():
        material = next(
            (item for item in materials if item.type.strip().lower() == name),
            None,
        )
        if not material:
            continue
        for status, config in targets.items():
            target_count = config.get("count", 0)
            ratio_hint = config.get(
                "target_ratio", 0.9 if status == "pending" else 0.8
            )
            existing = (
                db.query(TruckDelivery)
                .filter(
                    TruckDelivery.material_id == material.id,
                    TruckDelivery.status == status,
                )
                .count()
            )
            for _ in range(max(0, target_count - existing)):
                _add_future_delivery(
                    db,
                    materials,
                    status=status,
                    material=material,
                    target_ratio=ratio_hint,
                )


def _prepare_demo_delivery_queue() -> None:
    session = SessionLocal()
    try:
        materials = session.query(Material).all()
        queue: Deque[Tuple[int, float]] = deque()
        focus_plan = [
            ("sms clay", 0.97, 4),
            ("feldspar", 0.95, 4),
        ]
        for name, ratio, count in focus_plan:
            material = next(
                (item for item in materials if item.type.strip().lower() == name),
                None,
            )
            if not material:
                continue
            for _ in range(count):
                queue.append((material.id, ratio))
        DEMO_STATE["delivery_queue"] = queue
    finally:
        session.close()

@app.post("/api/demo/seed", response_model=List[MaterialRead])
def seed_demo_inventory(
    payload: DemoSeedRequest,
    db: Session = Depends(get_db),
) -> List[MaterialRead]:
    if not payload.inventory:
        raise HTTPException(status_code=400, detail="Inventory list cannot be empty")

    normalized_inventory = {}
    canonical_lookup = {}
    for item in payload.inventory:
        if not item.material.strip():
            continue
        canonical = canonical_material_name(item.material).strip()
        key = canonical.lower()
        normalized_inventory[key] = item
        canonical_lookup[key] = canonical

    if not normalized_inventory:
        raise HTTPException(status_code=400, detail="No valid material names supplied")

    materials_by_name = {
        material.type.strip().lower(): material
        for material in db.query(Material).all()
    }

    updated_materials: List[Material] = []

    for key, item in normalized_inventory.items():
        canonical = canonical_lookup[key]
        material = materials_by_name.get(key)
        if material:
            material.type = canonical
            material.weight = item.weight
            if item.humidity is not None:
                material.humidity = item.humidity
            if item.density is not None:
                material.density = item.density
            updated_materials.append(material)
        else:
            material = Material(
                type=canonical,
                weight=item.weight,
                humidity=item.humidity or 0.0,
                density=item.density or 0.0,
            )
            db.add(material)
            updated_materials.append(material)
            materials_by_name[key] = material

    db.flush()

    all_materials = list(db.query(Material).order_by(Material.id.asc()).all())
    shaped_materials = _shape_demo_levels(all_materials)
    unique_materials = {material.id: material for material in updated_materials}
    for material in shaped_materials:
        unique_materials.setdefault(material.id, material)
    updated_materials = list(unique_materials.values())
    db.flush()

    if payload.reset_deliveries:
        db.query(TruckDelivery).delete()
        db.flush()

    if payload.reset_history:
        db.query(MaterialInventoryHistory).delete()
        db.flush()
        _seed_demo_history(db, all_materials, replace=False)
    else:
        needs_history: List[Material] = []
        for material in updated_materials:
            existing_points = (
                db.query(MaterialInventoryHistory)
                .filter(MaterialInventoryHistory.material_id == material.id)
                .count()
            )
            if existing_points < 2:
                needs_history.append(material)

        if needs_history:
            _seed_demo_history(db, needs_history, replace=True)

        needs_history_ids = {material.id for material in needs_history}
        remaining = [
            material
            for material in updated_materials
            if material.id not in needs_history_ids
        ]
        if remaining:
            _record_history(db, remaining)

    _ensure_demo_deliveries(db, all_materials)

    db.commit()

    all_materials = db.query(Material).order_by(Material.type.asc()).all()
    return [MaterialRead.model_validate(_material_metrics(material)) for material in all_materials]


@app.post(
    "/api/simulation/demo",
    response_model=DemoStatus,
    status_code=202,
)
async def start_demo(duration_seconds: int = 30) -> DemoStatus:
    if _demo_running():
        raise HTTPException(status_code=409, detail="Demo already running")

    await _stop_demo()

    _prepare_demo_delivery_queue()

    start_time = datetime.utcnow()
    DEMO_STATE.update({"start_time": start_time, "duration": duration_seconds})
    DEMO_STATE["task"] = asyncio.create_task(_demo_runner(duration_seconds))

    return DemoStatus(
        running=True,
        seconds_remaining=float(duration_seconds),
        started_at=start_time,
        duration=duration_seconds,
    )


@app.post("/api/simulation/demo/stop", response_model=DemoStatus)
async def stop_demo() -> DemoStatus:
    await _stop_demo()
    return DemoStatus(running=False, seconds_remaining=None, started_at=None, duration=None)


@app.get("/api/simulation/demo/status", response_model=DemoStatus)
def demo_status() -> DemoStatus:
    running = _demo_running()
    return DemoStatus(
        running=running,
        seconds_remaining=_demo_seconds_remaining(),
        started_at=DEMO_STATE.get("start_time"),
        duration=DEMO_STATE.get("duration"),
    )


@app.get("/api/deliveries", response_model=List[TruckDeliveryRead])
def get_deliveries(db: Session = Depends(get_db)) -> List[TruckDeliveryRead]:
    return db.query(TruckDelivery).all()


@app.post("/api/deliveries", response_model=TruckDeliveryRead, status_code=201)
def create_delivery(
    payload: DeliveryCreateRequest,
    db: Session = Depends(get_db),
) -> TruckDeliveryRead:
    """Record a real truck delivery; wire this to your scale/TMS feed."""
    bol = BillOfLading(
        delivery_number=payload.delivery_num,
        po_number=None,
        customer=None,
        product=payload.material_code,
        net_weight_lb=payload.incoming_weight_lb,
        gross_weight_lb=None,
        tare_weight_lb=None,
        carrier=None,
        delivery_date=payload.delivery_time,
        material_code=payload.material_code,
    )
    delivery = _process_bill_of_lading(db, bol)
    if payload.status and delivery.status != payload.status:
        delivery.status = payload.status
        db.commit()
        db.refresh(delivery)
    return delivery


def _process_bill_of_lading(db: Session, payload: BillOfLading) -> TruckDelivery:
    material = (
        db.query(Material)
        .filter(func.lower(Material.type) == payload.material_code.lower())
        .one_or_none()
    )
    if not material:
        material = Material(
            type=payload.material_code,
            weight=0.0,
            humidity=0.0,
            density=0.0,
        )
        db.add(material)
        db.flush()  # assign id

    delivery = (
        db.query(TruckDelivery)
        .filter(TruckDelivery.delivery_num == payload.delivery_number)
        .one_or_none()
    )

    if delivery:
        delivery.incoming_weight = payload.net_weight_lb
        delivery.material_id = material.id
        delivery.delivery_time = payload.delivery_date.isoformat()
        delivery.status = delivery.status or "completed"
    else:
        delivery = TruckDelivery(
            delivery_num=payload.delivery_number,
            incoming_weight=payload.net_weight_lb,
            material_id=material.id,
            delivery_time=payload.delivery_date.isoformat(),
            status="completed",
        )
        db.add(delivery)

    material.apply_weight_change(payload.to_short_tons())
    _record_history(db, [material])

    db.commit()
    db.refresh(delivery)
    return delivery


@app.post(
    "/api/bol/import",
    response_model=TruckDeliveryRead,
    status_code=201,
)
def import_bill_of_lading(
    payload: BillOfLading,
    db: Session = Depends(get_db),
) -> TruckDeliveryRead:
    """Ingest a structured bill of lading payload and update material inventory."""
    delivery = _process_bill_of_lading(db, payload)
    return delivery


@app.post("/api/weight")
async def estimate_image_weight(file: UploadFile, density: float):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Invalid image file"}
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay, mass = calc_weight(image, density)
    _, buffer = cv2.imencode(".png", overlay)
    overlay_bytes = BytesIO(buffer.tobytes())
    return StreamingResponse(
        overlay_bytes,
        media_type="image/png",
        headers={"X-Mass-Short-Ton": f"{mass:.2f}"}
    )