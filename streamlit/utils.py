import ccxt, pandas as pd
from pathlib import Path
import json
from collections import deque
import pandas as pd

def create_history_buffer(df_full: pd.DataFrame, max_history: int):
    """
    Devuelve deque con los últimos `max_history` registros (cada elemento es dict o pd.Series).
    """
    dq = deque(maxlen=max_history)
    for _, row in df_full.tail(max_history).iterrows():
        dq.append(row.to_dict())
    return dq

def buffer_to_df(buffer: deque):
    import pandas as pd
    return pd.DataFrame(list(buffer))

def compute_online_features_from_buffer(buffer):
    df = buffer_to_df(buffer)
    # llama a la misma función que usas para features (asegúrate que sea causal)
    df_feat = fibo.add_technical_features(df['close'].values, high=df.get('high').values, low=df.get('low').values, volume=df.get('volume').values, window_long=50)
    return df_feat.iloc[-1]  # fila más reciente con features calculadas

def fetch_ccxt_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=500, exchange_id='binance', api_key=None, api_secret=None, sandbox=True):
    exchange_cls = getattr(ccxt, exchange_id)
    params = {"enableRateLimit": True}
    if api_key and api_secret:
        params["apiKey"] = api_key
        params["secret"] = api_secret
    exchange = exchange_cls(params)
    if sandbox:
        try:
            exchange.set_sandbox_mode(True)
        except Exception:
            pass
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=int(limit))
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def save_ohlcv_json(df, path):
    p = Path(path)
    if isinstance(df, pd.DataFrame):
        p.write_text(df.to_json(orient='records', date_format='iso'))
    else:
        p.write_text(pd.DataFrame({"close": list(df)}).to_json(orient='records'))

def load_ohlcv_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    import pandas as pd
    return pd.read_json(p)

import math
from pathlib import Path

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
    if all(c in df.columns for c in ['high','low','close']):
        df['atr'] = compute_atr(df, period=atr_period)
    else:
        df['atr'] = 0.0
    df['mom'] = df['close'].pct_change(periods=1).fillna(0.0)
    return df
