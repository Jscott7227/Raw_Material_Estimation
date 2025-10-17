from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class MaterialBase(BaseModel):
    type: str = Field(..., example="Super Strength 2")
    humidity: float = Field(0.0, ge=0.0)
    density: float = Field(0.0, ge=0.0)
    weight: float = Field(0.0, ge=0.0)


class MaterialCreate(MaterialBase):
    pass


class MaterialRead(MaterialBase):
    id: int

    model_config = ConfigDict(from_attributes=True)


class TruckDeliveryBase(BaseModel):
    delivery_num: str = Field(..., example="TRK-001")
    incoming_weight: float = Field(..., ge=0.0)
    material_id: int = Field(..., ge=1)
    delivery_time: str = Field(..., example="2025-10-16T15:20:00")
    status: str = Field("pending", pattern="^(pending|completed|upcoming)$")


class TruckDeliveryCreate(TruckDeliveryBase):
    status: Optional[str] = Field(default="pending")


class TruckDeliveryRead(TruckDeliveryBase):
    id: int

    model_config = ConfigDict(from_attributes=True)
