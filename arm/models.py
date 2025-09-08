import enum
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum as SQLAlchemyEnum
from sqlalchemy.orm import declarative_base
from pydantic import BaseModel, Field
from datetime import datetime

Base = declarative_base()

class OrderStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    FAILED = "failed"
    EXECUTED = "executed"

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    exchange_order_id = Column(String, unique=True, index=True, nullable=True)
    symbol = Column(String, index=True)
    type = Column(String)
    side = Column(String)
    amount = Column(Float)
    price = Column(Float, nullable=True)
    status = Column(SQLAlchemyEnum(OrderStatus), default=OrderStatus.OPEN)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class OrderCreate(BaseModel):
    symbol: str = Field(..., example="BTC/USDT")
    type: str = Field(..., example="limit")
    side: str = Field(..., example="buy")
    amount: float = Field(..., example=0.01)
    price: float | None = Field(None, example=50000.0)

class OrderResponse(BaseModel):
    id: int
    exchange_order_id: str | None
    symbol: str
    type: str
    side: str
    amount: float
    price: float | None
    status: OrderStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True