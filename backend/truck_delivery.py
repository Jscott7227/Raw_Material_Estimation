# app/models/truck_delivery.py
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from base import Base

class TruckDelivery(Base):
    __tablename__ = "truck_deliveries"

    id = Column(Integer, primary_key=True, index=True)
    delivery_num = Column(String, unique=True, nullable=False)  # e.g., TRK-001
    incoming_weight = Column(Float, nullable=False)             # total weight of the truck
    material_weights = Column(JSON, nullable=False)             # dict of {material_id: weight}
    delivery_time = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")                 # pending, completed, canceled
