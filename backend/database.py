# app/db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from material import Base, Material

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
