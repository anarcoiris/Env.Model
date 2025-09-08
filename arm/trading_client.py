import ccxt
import os
from dotenv import load_dotenv

class TradingClient:
    def __init__(self):
        load_dotenv()
        self.exchange_id = os.getenv('EXCHANGE_ID')
        self.api_key = os.getenv('EXCHANGE_API_KEY')
        self.secret_key = os.getenv('EXCHANGE_SECRET_KEY')
        self.sandbox_mode = os.getenv('EXCHANGE_SANDBOX_MODE', 'False').lower() in ('true', '1', 't')

        if not all([self.exchange_id, self.api_key, self.secret_key]):
            raise ValueError("Las credenciales del exchange no están configuradas en .env.arm")

        exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = exchange_class({
            'apiKey': self.api_key,
            'secret': self.secret_key,
        })

        if self.sandbox_mode:
            self.exchange.set_sandbox_mode(True)
        
        print(f"🤖 Cliente de Trading inicializado para '{self.exchange_id}'. Modo Sandbox: {self.sandbox_mode}")

    def create_order(self, symbol: str, type: str, side: str, amount: float, price: float = None):
        try:
            print(f"➡️  Intentando crear orden: {side} {amount} {symbol} @ {price if price else 'market'}")
            if type == 'limit' and price is None:
                raise ValueError("El precio es obligatorio para órdenes 'limit'")
            
            order = self.exchange.create_order(symbol, type, side, amount, price)
            return order
        except ccxt.BaseError as e:
            print(f"❌ Error de CCXT al crear la orden: {e}")
            return None

    def fetch_order(self, order_id: str, symbol: str):
        try:
            return self.exchange.fetch_order(order_id, symbol)
        except ccxt.BaseError as e:
            print(f"❌ Error de CCXT al obtener la orden {order_id}: {e}")
            return None
            
    def fetch_balance(self):
        try:
            return self.exchange.fetch_balance()
        except ccxt.BaseError as e:
            print(f"❌ Error de CCXT al obtener el balance: {e}")
            return None

trading_client = TradingClient()