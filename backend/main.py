from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import init_db

app = FastAPI()

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

async def lifespan(app: FastAPI):
    init_db()  # Create tables and preload materials
    yield

@app.get("/api/materials")
def get_materials():
    from database import SessionLocal
    from material import Material
    db = SessionLocal()
    materials = db.query(Material).all()
    db.close()
    return [{"id": m.id, "type": m.type, "weight": m.weight, "humidity": m.humidity} for m in materials]
