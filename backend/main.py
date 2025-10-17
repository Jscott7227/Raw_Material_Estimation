from contextlib import asynccontextmanager
from typing import Generator, List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import SessionLocal, init_db
from models import Material, TruckDelivery
from schemas import MaterialRead, TruckDeliveryCreate, TruckDeliveryRead

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # Create tables and preload materials
    yield

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


@app.get("/api/health")
def health_check():
    return {"status": "running"}

@app.get("/api/demo")
def demo():
    return {"message": "Hello from FastAPI backend!"}

@app.get("/api/materials", response_model=List[MaterialRead])
def get_materials(db: Session = Depends(get_db)) -> List[MaterialRead]:
    return db.query(Material).all()

@app.get("/api/deliveries", response_model=List[TruckDeliveryRead])
def get_deliveries(db: Session = Depends(get_db)) -> List[TruckDeliveryRead]:
    return db.query(TruckDelivery).all()

@app.post(
    "/api/truck_deliveries",
    response_model=TruckDeliveryRead,
    status_code=201,
)
def create_truck_delivery(
    payload: TruckDeliveryCreate,
    db: Session = Depends(get_db),
) -> TruckDeliveryRead:
    # Check for duplicate delivery_num
    if (
        db.query(TruckDelivery)
        .filter(TruckDelivery.delivery_num == payload.delivery_num)
        .first()
    ):
        raise HTTPException(status_code=400, detail="Delivery number already exists")

    delivery = TruckDelivery(
        delivery_num=payload.delivery_num,
        incoming_weight=payload.incoming_weight,
        material_id=payload.material_id,
        delivery_time=payload.delivery_time,
        status=payload.status or "pending",
    )

    db.add(delivery)
    db.commit()
    db.refresh(delivery)

    return delivery

@app.put(
    "/api/truck_deliveries/{delivery_id}/complete",
    response_model=TruckDeliveryRead,
)
def complete_truck_delivery(
    delivery_id: int,
    db: Session = Depends(get_db),
) -> TruckDeliveryRead:
    """Move a truck delivery from pending to completed and update material stock."""
    delivery = db.query(TruckDelivery).filter(TruckDelivery.id == delivery_id).first()

    if not delivery:
        raise HTTPException(status_code=404, detail="Delivery not found")

    if delivery.status != "pending":
        raise HTTPException(status_code=400, detail="Only pending deliveries can be completed")

    delivery.status = "completed"

    material = db.query(Material).filter(Material.id == delivery.material_id).first()
    if material:
        material.apply_weight_change(delivery.incoming_weight)

    db.commit()
    db.refresh(delivery)

    return delivery
