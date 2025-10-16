from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from database import init_db

async def lifespan(app: FastAPI):
    init_db()  # Create tables and preload materials
    yield

app = FastAPI(lifespan=lifespan)

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
            "material_weights": d.material_weights,
            "delivery_time": d.delivery_time.isoformat(),
            "status": d.status
        })
    
    return result