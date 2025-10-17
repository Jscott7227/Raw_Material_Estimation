# app/db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from models import Base, Material, TruckDelivery, MaterialInventoryHistory

DATABASE_URL = "sqlite:///./materials.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create tables if they do not exist"""
    Base.metadata.create_all(bind=engine)
    init_materials()  # Preload materials
    init_test_delivery()
    seed_inventory_history()
    
def init_test_delivery():
    """Add a temporary test truck delivery if none exists"""
    db = SessionLocal()
    
    # Check if any deliveries already exist
    if db.query(TruckDelivery).count() > 0:
        db.close()
        return

    # Example test delivery
    test_delivery = TruckDelivery(
        delivery_num="TRK-TEST-001",
        incoming_weight=1000,
        material_id=1,
        delivery_time=datetime.utcnow().isoformat(),
        status="pending",
    )

    db.add(test_delivery)
    db.commit()
    db.close()


def seed_inventory_history():
    """Ensure at least one inventory snapshot exists for each material"""
    db = SessionLocal()
    try:
        if db.query(MaterialInventoryHistory).count() > 0:
            return
        timestamp = datetime.utcnow()
        for material in db.query(Material).all():
            db.add(
                MaterialInventoryHistory(
                    material_id=material.id,
                    weight=material.weight,
                    recorded_at=timestamp,
                )
            )
        db.commit()
    finally:
        db.close()

def init_materials():
    """Initialize DB with 6 default materials with 0 weight and humidity"""
    db = SessionLocal()
    # Check if already initialized
    if db.query(Material).count() > 0:
        db.close()
        return

    material_names = ["Super Strength 2", "TN Stone", "SMS Clay", "Minspar", "Sandspar", "Feldspar", "LR28"]
    for name in material_names:
        mat = Material(type=name, humidity=0.0, density=0.0, weight=0.0)
        db.add(mat)
    db.commit()
    db.close()
