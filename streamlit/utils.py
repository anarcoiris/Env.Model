
import math
from pathlib import Path
import pandas as pd
import numpy as np

def quantize_price(price: float, tick: float):
    if tick is None or tick == 0:
        return float(price)
    return math.floor(price / tick) * tick

def quantize_amount(amount: float, lot: float):
    if lot is None or lot == 0:
        return float(amount)
    return math.floor(amount / lot) * lot

def compute_ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df: pd.DataFrame, period: int = 14):
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr

def compute_minimal_indicators(df: pd.DataFrame, window_fast: int = 10, window_slow: int = 50, rsi_period: int = 14, atr_period: int = 14):
    df = df.copy()
    if 'close' not in df.columns:
        raise ValueError('DataFrame must contain close column')
    df['ema_fast'] = compute_ema(df['close'], span=window_fast)
    df['ema_slow'] = compute_ema(df['close'], span=window_slow)
    df['rsi'] = compute_rsi(df['close'], period=rsi_period)
    # ATR needs high/low/close
    if all(c in df.columns for c in ['high', 'low', 'close']):
        df['atr'] = compute_atr(df, period=atr_period)
    else:
        df['atr'] = 0.0
    # simple momentum
    df['mom'] = df['close'].pct_change(periods=1).fillna(0.0)
    return df

def fetch_orderbook_top(orderbook, top_n=5):
    bids = orderbook.get('bids', [])[:top_n]
    asks = orderbook.get('asks', [])[:top_n]
    return {'bids': bids, 'asks': asks}
