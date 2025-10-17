from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Material(Base):
    __tablename__ = "materials"

    id: int = Column(Integer, primary_key=True, index=True)
    type: str = Column(String, nullable=False, unique=True)
    humidity: float = Column(Float, default=0.0)
    density: float = Column(Float, default=0.0)
    weight: float = Column(Float, default=0.0)

    deliveries = relationship("TruckDelivery", back_populates="material", cascade="all, delete-orphan")

    def calc_weight(self, volume: float) -> float:
        mass = self.density * volume
        self.weight = mass * 9.81
        return self.weight

    def update_humidity(self, humidity: float) -> float:
        self.humidity = humidity
        return self.humidity

    def apply_weight_change(self, delta: float) -> float:
        self.weight = max(self.weight + delta, 0)
        return self.weight


class TruckDelivery(Base):
    __tablename__ = "truck_deliveries"

    id: int = Column(Integer, primary_key=True, index=True)
    material_id: int = Column(Integer, ForeignKey("materials.id"), nullable=False)
    delivery_num: str = Column(String, unique=True, nullable=False)
    incoming_weight: float = Column(Float, nullable=False)
    delivery_time: str = Column(String, default=lambda: datetime.utcnow().isoformat())
    status: str = Column(String, default="pending")

    material = relationship("Material", back_populates="deliveries")

