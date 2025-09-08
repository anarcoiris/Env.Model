import os
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.orm import Session
from typing import List
from dotenv import load_dotenv

import models
import database
from trading_client import trading_client

app = FastAPI(
    title="🤖 El Brazo - API de Ejecución de Trading",
    description="API para ejecutar y gestionar órdenes en el mercado.",
    version="1.0.0"
)

# --- Seguridad por API Key ---
load_dotenv()
API_KEY = os.getenv("ARM_API_KEY")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(key: str = Security(api_key_header)):
    if key == API_KEY:
        return key
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Credenciales no válidas o ausentes."
        )

@app.on_event("startup")
def on_startup():
    database.create_db_and_tables()
    print("🚀 La base de datos y las tablas han sido creadas/verificadas.")
    print("API 'El Brazo' iniciada y lista para recibir órdenes.")

@app.get("/", tags=["Estado"])
def read_root():
    return {"message": "El Brazo está operativo. Listo para ejecutar órdenes."}

@app.post("/order", response_model=models.OrderResponse, tags=["Trading"], dependencies=[Depends(get_api_key)])
def create_order(order: models.OrderCreate, db: Session = Depends(database.get_db)):
    print(f"📬 Petición de orden recibida: {order.model_dump_json()}")
    
    exchange_order = trading_client.create_order(
        symbol=order.symbol, type=order.type, side=order.side,
        amount=order.amount, price=order.price
    )

    if not exchange_order or 'id' not in exchange_order:
        raise HTTPException(status_code=500, detail="Fallo al crear la orden en el exchange.")

    db_order = models.Order(
        exchange_order_id=exchange_order.get('id'), symbol=order.symbol, type=order.type,
        side=order.side, amount=order.amount, price=order.price,
        status=models.OrderStatus.OPEN
    )
    db.add(db_order)
    db.commit()
    db.refresh(db_order)
    
    print(f"✅ Orden creada y registrada en la DB. ID Local: {db_order.id}, ID Exchange: {db_order.exchange_order_id}")
    return db_order

@app.get("/orders", response_model=List[models.OrderResponse], tags=["Reportes"], dependencies=[Depends(get_api_key)])
def read_orders(skip: int = 0, limit: int = 100, db: Session = Depends(database.get_db)):
    return db.query(models.Order).offset(skip).limit(limit).all()

@app.get("/balance", tags=["Reportes"], dependencies=[Depends(get_api_key)])
def get_balance():
    balance = trading_client.fetch_balance()
    if balance is None:
        raise HTTPException(status_code=500, detail="No se pudo obtener el balance desde el exchange.")
    return balance