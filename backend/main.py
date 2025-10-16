from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from database import init_db
from datetime import datetime

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

@app.get("/api/health")
def health_check():
    return {"status": "running"}

@app.get("/api/demo")
def demo():
    return {"message": "Hello from FastAPI backend!"}

@app.get("/api/materials")
def get_materials():
    from database import SessionLocal
    from material import Material
    db = SessionLocal()
    materials = db.query(Material).all()
    db.close()
    return [{"id": m.id, "type": m.type, "weight": m.weight, "humidity": m.humidity} for m in materials]

@app.get("/api/deliveries")
def get_deliveries():
    from database import SessionLocal
    from truck_delivery import TruckDelivery
    db = SessionLocal()
    deliveries = db.query(TruckDelivery).all()
    db.close()
    result = []
    for d in deliveries:
        result.append({
            "id": d.id,
            "delivery_num": d.delivery_num,
            "incoming_weight": d.incoming_weight,
            "material_id": d.material_id,
            "delivery_time": d.delivery_time,
            "status": d.status
        })
    
    return result

@app.post("/api/truck_deliveries")
def create_truck_delivery(payload: dict = Body(...)):
    from database import SessionLocal
    from truck_delivery import TruckDelivery
    """
    Create a new truck delivery with a single material.
    JSON body example:
    {
        "delivery_num": "TRK-002",
        "incoming_weight": 500,
        "material_id": 3,
        "delivery_time": "2025-10-16T15:20:00",
    }
    Status is automatically set to 'pending'.
    """
    required_keys = ["delivery_num", "incoming_weight", "material_id", "delivery_time"]
    for key in required_keys:
        if key not in payload:
            raise HTTPException(status_code=400, detail=f"Missing '{key}' in request body")

    db = SessionLocal()

    # Check for duplicate delivery_num
    if db.query(TruckDelivery).filter(TruckDelivery.delivery_num == payload["delivery_num"]).first():
        db.close()
        raise HTTPException(status_code=400, detail="Delivery number already exists")

    delivery = TruckDelivery(
        delivery_num=payload["delivery_num"],
        incoming_weight=payload["incoming_weight"],
        material_id=payload["material_id"],
        delivery_time=payload["delivery_time"],
        status="pending"
    )

    db.add(delivery)
    db.commit()
    db.refresh(delivery)
    db.close()

    return {
        "id": delivery.id,
        "delivery_num": delivery.delivery_num,
        "incoming_weight": delivery.incoming_weight,
        "material_id": delivery.material_id,
        "delivery_time": delivery.delivery_time,
        "status": delivery.status
    }

@app.put("/api/truck_deliveries/{delivery_id}/complete")
def complete_truck_delivery(delivery_id: int):
    from database import SessionLocal
    from truck_delivery import TruckDelivery
    from material import Material
    """
    Move a truck delivery from pending to completed.
    Apply the incoming_weight to the Material stock.
    """
    db = SessionLocal()
    delivery = db.query(TruckDelivery).filter(TruckDelivery.id == delivery_id).first()
    
    if not delivery:
        db.close()
        raise HTTPException(status_code=404, detail="Delivery not found")
    
    if delivery.status != "pending":
        db.close()
        raise HTTPException(status_code=400, detail="Only pending deliveries can be completed")
    
    # Mark as completed
    delivery.status = "completed"
    
    # Optionally: update the Material weight
    material = db.query(Material).filter(Material.id == delivery.material_id).first()
    if material:
        material.apply_weight_change(delivery.incoming_weight)
    
    db.commit()
    db.refresh(delivery)
    db.close()
    
    return {
        "id": delivery.id,
        "delivery_num": delivery.delivery_num,
        "status": delivery.status,
        "material_id": delivery.material_id,
        "incoming_weight": delivery.incoming_weight
    }


