from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MaterialBase(BaseModel):
    type: str = Field(...)
    humidity: float = Field(0.0, ge=0.0)
    density: float = Field(0.0, ge=0.0)
    weight: float = Field(0.0, ge=0.0)


class MaterialCreate(MaterialBase):
    pass


class MaterialUpdate(BaseModel):
    type: Optional[str] = None
    weight: Optional[float] = Field(default=None, ge=0)
    humidity: Optional[float] = Field(default=None, ge=0)
    density: Optional[float] = Field(default=None, ge=0)


class MaterialRead(MaterialBase):
    id: int
    bin_capacity: float = Field(1200.0, ge=0)
    fill_ratio: float = Field(..., ge=0)
    bins_filled: float = Field(..., ge=0)
    needs_reorder: bool
    consumption_std_tons: float = Field(0.0, ge=0)
    delivery_std_tons: float = Field(0.0, ge=0)
    open_orders_tons: float = Field(0.0, ge=0)

    model_config = ConfigDict(from_attributes=True)


class TruckDeliveryBase(BaseModel):
    delivery_num: str = Field(...)
    incoming_weight: float = Field(..., ge=0.0)
    material_id: int = Field(..., ge=1)
    delivery_time: datetime = Field(...)
    status: str = Field("upcoming", pattern="^(completed|upcoming)$")


class TruckDeliveryRead(TruckDeliveryBase):
    id: int

    model_config = ConfigDict(from_attributes=True)


class SensorEvent(BaseModel):
    material_id: int = Field(..., ge=1)
    material_type: str = Field(...)
    delta: float = Field(..., le=0)
    remaining_weight: float = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=_utc_now)


class MaterialAlert(BaseModel):
    material_id: int
    material_type: str
    weight: float
    fill_ratio: float
    bins_filled: float
    alert_level: str
    message: str


class DemoStatus(BaseModel):
    running: bool
    seconds_remaining: Optional[float] = None
    started_at: Optional[datetime] = None
    duration: Optional[int] = None


class MaterialRecommendation(BaseModel):
    material_id: int
    material_type: str
    current_weight: float
    average_daily_consumption: float
    projected_weight_seven_days: float
    days_until_reorder: Optional[float]
    recommended_order_date: Optional[datetime]
    recommended_order_tons: Optional[float]
    rationale: str
    pending_orders_tons: float = Field(0.0, ge=0)
    upcoming_orders: List["OrderSummary"] = Field(default_factory=list)


class DemoInventoryItem(BaseModel):
    material: str
    weight: float = Field(..., ge=0)
    humidity: Optional[float] = Field(default=None, ge=0)
    density: Optional[float] = Field(default=None, ge=0)


class DemoSeedRequest(BaseModel):
    inventory: List[DemoInventoryItem]
    reset_deliveries: bool = False
    reset_history: bool = True

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inventory": [
                    {"material": "Super Strength 2", "weight": 950},
                    {"material": "TN Stone", "weight": 620},
                    {"material": "SMS Clay", "weight": 480},
                    {"material": "Minspar", "weight": 810},
                    {"material": "Sandspar", "weight": 585},
                    {"material": "Feldspar", "weight": 720},
                    {"material": "LR28", "weight": 540}
                ],
                "reset_deliveries": True,
                "reset_history": True
            }
        }
    )


class MaterialAdjustRequest(BaseModel):
    delta: float
    reason: Optional[str] = None


class BillOfLading(BaseModel):
    delivery_number: str
    po_number: Optional[str] = None
    customer: Optional[str] = None
    product: str
    net_weight_lb: float = Field(..., ge=0)
    gross_weight_lb: Optional[float] = Field(default=None, ge=0)
    tare_weight_lb: Optional[float] = Field(default=None, ge=0)
    carrier: Optional[str] = None
    delivery_date: datetime
    material_code: str

    model_config = ConfigDict(from_attributes=True)

    def to_short_tons(self) -> float:
        return round(self.net_weight_lb / 2000, 2)


class DeliveryCreateRequest(BaseModel):
    delivery_num: str
    material_code: str
    incoming_weight_lb: float = Field(..., ge=0)
    delivery_time: datetime
    status: Optional[str] = Field(default="completed", pattern="^(completed|upcoming)$")

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Optional[str]) -> str:
        from .utils import normalize_delivery_status

        if value is None:
            return "completed"
        return normalize_delivery_status(value, default="completed")


class OrderSummary(BaseModel):
    order_id: str
    material: str
    requested_date: datetime
    required_tons: float
    status: str


MaterialRecommendation.model_rebuild()


class BillOfLadingSimulationConfig(BaseModel):
    min_trucks_per_day: int = Field(15, gt=0)
    max_trucks_per_day: int = Field(30, gt=0)
    min_tons: float = Field(20.0, gt=0)
    max_tons: float = Field(35.0, gt=0)

    @model_validator(mode="after")
    def _check_ranges(self) -> "BillOfLadingSimulationConfig":
        if self.min_trucks_per_day > self.max_trucks_per_day:
            raise ValueError("min_trucks_per_day must be <= max_trucks_per_day")
        if self.min_tons > self.max_tons:
            raise ValueError("min_tons must be <= max_tons")
        return self
