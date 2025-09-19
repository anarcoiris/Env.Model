import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import math
import time

@dataclass
class Trade:
    ts: str
    symbol: str
    side: str  # 'buy'|'sell'
    price: float
    size: float
    fee: float
    slippage: float
    pnl: float = 0.0
    filled: bool = True

class PaperTrader:
    def __init__(self, initial_balance: float = 10000.0, commission_pct: float = 0.0005, slippage_model: str = "sqrt", liquidity: float = 1.0):
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.equity = float(initial_balance)
        self.positions = {}  # symbol -> {'size':..., 'avg_price':...}
        self.trades = []     # list of trade dicts
        self.commission_pct = float(commission_pct)
        self.slippage_model = slippage_model
        self.liquidity = float(liquidity)

    def _calc_slippage(self, size, price):
        if self.slippage_model == "fixed":
            return price * 0.0001
        # sqrt model
        return price * (0.0001 * math.sqrt(max(size, 1.0)) / max(self.liquidity, 0.0001))

    def execute_order(self, symbol: str, side: str, price: float, size: float):
        slippage = self._calc_slippage(size, price)
        exec_price = price + (slippage if side == "buy" else -slippage)
        fee = abs(exec_price * size) * self.commission_pct
        # update position (simple netting)
        pos = self.positions.get(symbol, {"size": 0.0, "avg_price": 0.0})
        if side == "buy":
            new_size = pos["size"] + size
            new_avg = (pos["avg_price"] * pos["size"] + exec_price * size) / new_size if new_size != 0 else exec_price
            pos["size"] = new_size
            pos["avg_price"] = new_avg
            self.balance -= (exec_price * size) + fee
        else:
            # sell - reduce position
            new_size = pos["size"] - size
            pnl = 0.0
            if size > 0 and pos["size"] > 0:
                pnl = (exec_price - pos["avg_price"]) * min(size, pos["size"])
                self.balance += (exec_price * size) - fee
            else:
                # short or closing beyond current - treat as cash trade
                self.balance += (exec_price * size) - fee
            pos["size"] = new_size
            # avg_price remains if still open; if closed reset
            if pos["size"] == 0:
                pos["avg_price"] = 0.0
        self.positions[symbol] = pos
        self._update_equity(symbol, market_price=exec_price)
        tr = Trade(ts=str(pd.Timestamp.utcnow()), symbol=symbol, side=side, price=exec_price, size=size, fee=fee, slippage=slippage)
        self.trades.append(asdict(tr))
        return tr

    def _update_equity(self, symbol, market_price):
        # compute unrealized pnl across positions
        total = self.balance
        for s, p in self.positions.items():
            total += p["size"] * market_price if p["size"] != 0 else 0.0
        self.equity = total

    def close_all(self, symbol, market_price):
        pos = self.positions.get(symbol)
        if not pos or pos["size"] == 0:
            return
        size = abs(pos["size"])
        side = "sell" if pos["size"] > 0 else "buy"
        return self.execute_order(symbol, side, market_price, size)