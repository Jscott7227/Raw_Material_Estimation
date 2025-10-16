# app/models/material.py
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from base import Base

class Material(Base):
    __tablename__ = "materials"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, nullable=False)
    humidity = Column(Float, default=0.0)
    density = Column(Float, default=0.0)
    weight = Column(Float, default=0.0)

    def calc_weight(self, volume: float):
        mass = self.density * volume
        self.weight = mass * 9.81
        return self.weight

    def update_humidity(self, humidity: float):
        self.humidity = humidity
        return self.humidity

    def apply_weight_change(self, delta: float):
        self.weight = max(self.weight + delta, 0)
        return self.weight
