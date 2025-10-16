# app/models/truck_delivery.py
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from base import Base

class TruckDelivery(Base):
    __tablename__ = "truck_deliveries"

    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("materials.id"), nullable=False)
    delivery_num = Column(String, unique=True, nullable=False)  # e.g., TRK-001
    incoming_weight = Column(Float, nullable=False)             # total weight of the truck
    delivery_time = Column(String, default=datetime.utcnow)
    status = Column(String, default="pending")                 # pending, completed, canceled
