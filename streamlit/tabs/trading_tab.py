# tabs/trading_tab.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import time, math, json, threading
import importlib
from datetime import datetime, timedelta

# optional imports
try:
    import ccxt
except Exception:
    ccxt = None

try:
    import joblib
except Exception:
    joblib = None

try:
    import torch
except Exception:
    torch = None

# Import fiboevo functions/models (fallback to import_module)
try:
    import fiboevo as fibo
except Exception:
    fibo = importlib.import_module("fiboevo")

LEDGER_PATH = Path("trades_ledger.csv")
BOT_STATE_PATH = Path("trading_bot_state.json")  # optional persistence of some state


# ---------------------------
# Utility helpers
# ---------------------------
def timeframe_to_seconds(tf: str):
    """Convierte timeframe tipo '1m','5m','1h','1d' a segundos."""
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    # fallback assume minutes
    try:
        return int(tf) * 60
    except Exception:
        return 60


def get_exchange(exchange_id: str, api_key: str = None, api_secret: str = None, sandbox: bool = True):
    if ccxt is None:
        raise RuntimeError("ccxt no está instalado. Instálalo con `pip install ccxt` para usar exchanges.")
    cls = getattr(ccxt, exchange_id, None)
    if cls is None:
        raise RuntimeError(f"Exchange id {exchange_id} no reconocido por ccxt.")
    params = {"enableRateLimit": True}
    if api_key and api_secret:
        params.update({"apiKey": api_key, "secret": api_secret})
    ex = cls(params)
    if sandbox:
        # Some exchanges implement set_sandbox_mode
        try:
            ex.set_sandbox_mode(True)
        except Exception:
            pass
    return ex


def get_market_precision(exchange, symbol: str):
    """
    Intenta extraer precision/step del mercado desde ccxt market metadata.
    Devuelve (tick, lot) o (None,None) si no disponible.
    """
    try:
        markets = exchange.load_markets()
        m = markets.get(symbol.replace("/", ""), None) or markets.get(symbol, None)
        if m is None:
            # try case-insensitive search
            for k, v in markets.items():
                if k.lower() == symbol.lower() or ("/" in k and k.replace("/", "").lower() == symbol.replace("/", "").lower()):
                    m = v
                    break
        if m is None:
            return None, None

        tick = None
        lot = None
        if isinstance(m.get("precision"), dict):
            tick = m["precision"].get("price", None)
            lot = m["precision"].get("amount", None)
        if tick is None and isinstance(m.get("limits"), dict):
            if isinstance(m["limits"].get("price"), dict):
                tick = m["limits"]["price"].get("min", None)
            if isinstance(m["limits"].get("amount"), dict):
                lot = m["limits"]["amount"].get("min", None)

        if tick is not None and tick > 0:
            if tick < 1 and tick > 0:
                return float(tick), float(lot) if lot else None
        return (float(tick) if tick else None, float(lot) if lot else None)
    except Exception:
        return None, None


def quantize_price(price: float, tick: float):
    if tick is None or tick == 0:
        return float(price)
    return math.floor(price / tick) * tick


def quantize_amount(amount: float, lot: float):
    if lot is None or lot == 0:
        return float(amount)
    return math.floor(amount / lot) * lot


def read_ledger():
    if LEDGER_PATH.exists():
        try:
            return pd.read_csv(LEDGER_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def append_ledger(entry: dict):
    """Guarda una fila en el ledger; añade columnas de posición/equity si faltan."""
    df = read_ledger()
    # use concat instead of deprecated append
    df_new = pd.concat([df, pd.DataFrame([entry])], ignore_index=True, sort=False)
    df_new.to_csv(LEDGER_PATH, index=False)


# ---------------------------
# Model & data utilities
# ---------------------------
def create_history_buffer_from_df(df_full: pd.DataFrame, max_history: int):
    """Devuelve DataFrame con las últimas max_history filas (en orden ascendente)."""
    if df_full is None or len(df_full) == 0:
        return pd.DataFrame()
    return df_full.tail(max_history).reset_index(drop=True)


def prepare_model_input(df_segment: pd.DataFrame, feature_cols: list, seq_len: int, scaler_path: str = None):
    """
    Prepara un tensor numpy 3D (N=1, seq_len, features) para alimentar el modelo.
    - df_segment: DataFrame ordenado ascendentemente, con al menos seq_len filas.
    """
    if df_segment is None or len(df_segment) < seq_len:
        raise ValueError(f"Insufficient history: need >= seq_len ({seq_len}) rows, got {len(df_segment) if df_segment is not None else 0}")

    seg = df_segment.iloc[-seq_len:].copy().reset_index(drop=True)
    X = seg[feature_cols].values.astype(np.float32)

    if scaler_path and Path(scaler_path).exists() and joblib is not None:
        try:
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X)
        except Exception as e:
            st.warning(f"Fallo al cargar/transformar con scaler: {e} — se usará X sin escalar.")
    return np.expand_dims(X, axis=0)


def predict_with_model(model_path: str, X_numpy):
    """
    Carga modelo desde model_path usando fibo.load_model (si existe) o cargando torch directly.
    Devuelve una diccionario de outputs según fiboevo.predict API (supuesto: retorno & vol).
    """
    device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
    if not Path(model_path).exists():
        raise FileNotFoundError("Model path not found: " + str(model_path))

    try:
        model = fibo.load_model(model_path, input_size=X_numpy.shape[2], device=device)
    except Exception:
        try:
            model = fibo.LSTM2Head(input_size=X_numpy.shape[2], hidden_size=64)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar el modelo con fibo.load_model ni fallback: {e}")

    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_numpy).to(device)
        out_ret, out_vol = model(xt)
        try:
            ret = out_ret.cpu().numpy().ravel()[0]
        except Exception:
            ret = float(out_ret.cpu().numpy().ravel()[0])
        try:
            vol = out_vol.cpu().numpy().ravel()[0]
        except Exception:
            vol = float(out_vol.cpu().numpy().ravel()[0])
    return {"pred_ret": float(ret), "pred_vol": float(vol)}


# ---------------------------
# Order execution (paper and live)
# (re-using your original execute_order with small tweaks)
# ---------------------------
def execute_order(exchange, symbol: str, side: str, amount: float, price: float, order_type: str = "limit",
                  paper: bool = True, slippage_tolerance_pct: float = 0.5, tick: float = None, lot: float = None):
    """
    Ejecuta la orden en modo paper (simulado) o real (ccxt exchange).
    Retorna un dic con info del fill.
    """
    now_ts = pd.Timestamp.utcnow().isoformat()
    if paper:
        executed_price = price
        slippage = price * (slippage_tolerance_pct / 100.0)
        if side.lower() == "buy":
            executed_price = price + slippage
        else:
            executed_price = price - slippage
        if tick:
            executed_price = quantize_price(executed_price, tick)
        if lot:
            amount = quantize_amount(amount, lot)
        out = {
            "timestamp": now_ts,
            "mode": "paper",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "price": float(price),
            "executed_price": float(executed_price),
            "amount": float(amount),
            "status": "filled",
        }
        append_ledger(out)
        return out

    # live order via ccxt
    try:
        if tick:
            price_q = quantize_price(price, tick)
        else:
            price_q = price
        if lot:
            amount_q = quantize_amount(amount, lot)
        else:
            amount_q = amount

        params = {}
        if order_type == "market":
            order = exchange.create_market_order(symbol, side, amount_q, params)
        else:
            order = exchange.create_limit_order(symbol, side, amount_q, price_q, params)
        out = {
            "timestamp": now_ts,
            "mode": "live",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "price": float(price_q),
            "amount": float(amount_q),
            "status": order,
        }
        append_ledger(out)
        return out
    except Exception as e:
        raise RuntimeError(f"Live order failed: {e}")


# ---------------------------
# TradingBot class
# ---------------------------
class TradingBot:
    """
    Clase que encapsula estado y lógica de trading en modo continuo (no bloqueante).
    - tick(): ejecutar un "paso" si corresponde (nueva vela/tick disponible).
    - start()/stop() para controlar ejecución (via session_state en streamlit).
    """

    def __init__(self, *,
                 model_path: str,
                 scaler_path: str,
                 feature_cols: list,
                 seq_len: int,
                 horizon: int,
                 exchange_id: str,
                 symbol: str,
                 timeframe: str,
                 api_key: str = None,
                 api_secret: str = None,
                 paper: bool = True,
                 max_pos_pct: float = 2.0,
                 max_daily_loss: float = 100.0,
                 max_orders_per_min: int = 6,
                 slippage_tolerance_pct: float = 0.5,
                 tick_rounding: bool = True):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.horizon = horizon
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.max_pos_pct = max_pos_pct
        self.max_daily_loss = max_daily_loss
        self.max_orders_per_min = max_orders_per_min
        self.slippage_tolerance_pct = slippage_tolerance_pct
        self.tick_rounding = tick_rounding

        self.exchange = None
        self.tick_seconds = timeframe_to_seconds(timeframe)

        # state
        self.running = False
        self.last_fetch_ts = None  # timestamp of last candle in seconds
        self.position = {"side": "flat", "entry_price": None, "amount": 0.0}
        self.equity = 10000.0  # default paper equity (can be overriden)
        self.initial_equity = self.equity
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.order_timestamps = []  # for rate limiting
        self.logs = []
        self.lock = threading.Lock()  # protección para multithread (por si acaso)

        # load exchange early if possible
        if ccxt is not None:
            try:
                self.exchange = get_exchange(exchange_id, api_key, api_secret, sandbox=self.paper)
            except Exception as e:
                # keep as None but log
                self.log(f"Warning: no exchange available: {e}")

        # obtain market tick/lot if possible
        self.tick, self.lot = (None, None)
        if self.exchange:
            try:
                self.tick, self.lot = get_market_precision(self.exchange, symbol)
            except Exception as e:
                self.log(f"Could not get market precision: {e}")

        # Try to reconstruct state from ledger (paper)
        self._reconstruct_from_ledger()

    # ---- logging helpers ----
    def log(self, msg: str):
        ts = pd.Timestamp.utcnow().isoformat()
        row = f"{ts} | {msg}"
        self.logs.append(row)
        # keep logs trimmed
        if len(self.logs) > 2000:
            self.logs = self.logs[-1000:]

    def _reconstruct_from_ledger(self):
        """Reconstruir posición/equity a partir del ledger si existe (paper)."""
        try:
            ledger = read_ledger()
            if ledger is None or len(ledger) == 0:
                return
            # try to find last open position; simple heuristic:
            # build equity starting from initial equity
            eq = self.equity
            last_pos = {"side": "flat", "entry_price": None, "amount": 0.0}
            for _, row in ledger.iterrows():
                # if it's paper, it should have executed_price
                side = str(row.get("side", "")).lower()
                executed_price = float(row.get("executed_price", row.get("price", np.nan)))
                amount = float(row.get("amount", 0.0))
                mode = row.get("mode", "paper")
                if side in ("buy", "sell") and mode == "paper":
                    # naive PnL accounting:
                    if last_pos["side"] == "flat":
                        # open position
                        last_pos = {"side": side, "entry_price": executed_price, "amount": amount}
                    else:
                        # close/flip: compute pnl
                        if last_pos["side"] == "buy" and side == "sell":
                            pnl = (executed_price - last_pos["entry_price"]) * last_pos["amount"]
                        elif last_pos["side"] == "sell" and side == "buy":
                            pnl = (last_pos["entry_price"] - executed_price) * last_pos["amount"]
                        else:
                            pnl = 0.0
                        eq += pnl
                        last_pos = {"side": "flat", "entry_price": None, "amount": 0.0}
            self.equity = float(eq)
            self.initial_equity = float(eq)
            self.position = last_pos
            self.log("Reconstructed state from ledger: equity={:.2f}, position={}".format(self.equity, self.position))
        except Exception as e:
            self.log(f"Reconstruct ledger failed: {e}")

    # ---- rate limiting ----
    def _can_send_order(self):
        now = time.time()
        # drop timestamps older than 60s
        self.order_timestamps = [t for t in self.order_timestamps if now - t < 60.0]
        return len(self.order_timestamps) < self.max_orders_per_min

    def _record_order_timestamp(self):
        self.order_timestamps.append(time.time())

    # ---- position helpers ----
    def _open_position(self, side: str, amount: float, price: float):
        """Abrir posición: registra orden y actualiza estado paper."""
        if not self._can_send_order():
            raise RuntimeError("Rate limit: too many orders in last minute")

        # create execution
        out = execute_order(
            exchange=self.exchange,
            symbol=self.symbol,
            side=side,
            amount=amount,
            price=price,
            order_type="market",
            paper=self.paper,
            slippage_tolerance_pct=self.slippage_tolerance_pct,
            tick=self.tick if self.tick_rounding else None,
            lot=self.lot if self.tick_rounding else None
        )
        self._record_order_timestamp()

        # update internal state / ledger extended fields
        executed_price = float(out.get("executed_price", out.get("price", price)))
        # if opening from flat
        if self.position["side"] == "flat":
            self.position = {"side": side, "entry_price": executed_price, "amount": float(out.get("amount", amount))}
            self.log(f"Opened {side} pos amount={self.position['amount']:.6f} at {executed_price:.2f}")
            # augment ledger entry with position after and equity snapshot
            out.update({
                "position_after": self.position["side"],
                "equity": self.equity,
                "pnl_realized": 0.0
            })
            append_ledger(out)
            return out

        # if flipping side: close existing then open new
        prev = dict(self.position)
        # compute pnl closing prev with this executed_price
        if prev["side"] == "buy" and side == "sell":
            pnl = (executed_price - prev["entry_price"]) * prev["amount"]
        elif prev["side"] == "sell" and side == "buy":
            pnl = (prev["entry_price"] - executed_price) * prev["amount"]
        else:
            pnl = 0.0
        self.realized_pnl += pnl
        self.equity += pnl
        # record close trade (we already recorded one via execute_order - but for clarity augment)
        out.update({
            "position_after": side,
            "equity": self.equity,
            "pnl_realized": pnl
        })
        # set new position as side (we assume market flip same amount)
        self.position = {"side": side, "entry_price": executed_price, "amount": float(out.get("amount", amount))}
        append_ledger(out)
        self.log(f"Flipped position {prev['side']} -> {side}, realized pnl {pnl:.2f}, equity {self.equity:.2f}")
        return out

    def _close_position(self, price: float):
        """Cerrar posición actual al precio dado."""
        if self.position["side"] == "flat":
            self.log("No position to close.")
            return None

        side = "sell" if self.position["side"] == "buy" else "buy"  # closing side
        amount = self.position["amount"]
        if not self._can_send_order():
            raise RuntimeError("Rate limit: too many orders in last minute")

        out = execute_order(
            exchange=self.exchange,
            symbol=self.symbol,
            side=side,
            amount=amount,
            price=price,
            order_type="market",
            paper=self.paper,
            slippage_tolerance_pct=self.slippage_tolerance_pct,
            tick=self.tick if self.tick_rounding else None,
            lot=self.lot if self.tick_rounding else None
        )
        self._record_order_timestamp()
        executed_price = float(out.get("executed_price", out.get("price", price)))

        # compute pnl
        if self.position["side"] == "buy":
            pnl = (executed_price - self.position["entry_price"]) * self.position["amount"]
        else:
            pnl = (self.position["entry_price"] - executed_price) * self.position["amount"]
        self.realized_pnl += pnl
        self.equity += pnl

        out.update({
            "position_after": "flat",
            "equity": self.equity,
            "pnl_realized": pnl
        })
        append_ledger(out)
        self.log(f"Closed position {self.position['side']} at {executed_price:.2f}, pnl={pnl:.2f}, equity={self.equity:.2f}")
        self.position = {"side": "flat", "entry_price": None, "amount": 0.0}
        return out

    # ---- core tick logic ----
    def tick(self, history_df: pd.DataFrame, enter_threshold: float, exit_threshold: float, position_pct: float):
        """
        Ejecutar un tick: si hay nueva vela (comparando last_fetch_ts con última de history_df),
        hacer predict y decidir órdenes.
        - history_df: DataFrame ordenado ascendentemente (timestamp), con al menos seq_len
        - position_pct: % de equity a usar para nueva posición
        """
        with self.lock:
            try:
                if history_df is None or len(history_df) < self.seq_len:
                    return {"status": "no_data"}
                # detect last candle ts (as seconds)
                last_ts = int(pd.to_datetime(history_df.iloc[-1]["timestamp"]).timestamp())
                if self.last_fetch_ts is not None and last_ts <= self.last_fetch_ts:
                    # no new candle
                    return {"status": "no_new_candle"}
                # ensure spacing by timeframe (prevents multiple ticks on same candle if rerun quickly)
                now = time.time()
                if self.last_fetch_ts is not None and (last_ts - self.last_fetch_ts) < (self.tick_seconds * 0.5):
                    # too soon (partial candle), skip
                    self.last_fetch_ts = last_ts
                    return {"status": "partial_candle_skipped"}
                # proceed: prepare input & predict
                X = prepare_model_input(history_df, self.feature_cols, self.seq_len, scaler_path=self.scaler_path)
                pred = predict_with_model(self.model_path, X)
                pred_ret = float(pred.get("pred_ret", 0.0))
                pred_vol = float(pred.get("pred_vol", 0.0))
                self.log(f"Tick: pred_ret={pred_ret:.6f}, pred_vol={pred_vol:.6f}")
                # decide signal
                signal = "hold"
                if pred_ret > enter_threshold:
                    signal = "buy"
                elif pred_ret < exit_threshold:
                    signal = "sell"

                last_close = float(history_df.iloc[-1]["close"])
                # position sizing
                size_usd = self.equity * (position_pct / 100.0)
                amount = size_usd / last_close
                # enforce lot rounding if configured
                if self.tick_rounding and self.lot:
                    amount = quantize_amount(amount, self.lot)
                # no-op for zero amount
                if amount <= 0:
                    self.log("Computed amount <= 0, skipping.")
                    return {"status": "zero_amount"}

                # Decision FSM
                action = None
                out = None
                if self.position["side"] == "flat":
                    if signal in ("buy", "sell"):
                        # open new pos
                        action = f"open_{signal}"
                        try:
                            out = self._open_position(signal, amount, last_close)
                        except Exception as e:
                            self.log(f"Open position failed: {e}")
                elif self.position["side"] == "buy":
                    if signal == "sell":
                        # flip: close and open sell
                        action = "flip_buy_to_sell"
                        try:
                            out = self._open_position("sell", amount, last_close)
                        except Exception as e:
                            self.log(f"Flip failed: {e}")
                    elif signal == "hold":
                        action = "hold_buy"
                    elif signal == "buy":
                        action = "add_buy"  # optional scaling in — not implemented
                elif self.position["side"] == "sell":
                    if signal == "buy":
                        action = "flip_sell_to_buy"
                        try:
                            out = self._open_position("buy", amount, last_close)
                        except Exception as e:
                            self.log(f"Flip failed: {e}")
                    elif signal == "hold":
                        action = "hold_sell"
                    elif signal == "sell":
                        action = "add_sell"  # optional scaling in

                # update last_fetch_ts
                self.last_fetch_ts = last_ts

                # Safety checks: daily loss
                # Very simple: if equity dropped below initial - max_daily_loss, stop trading
                if (self.initial_equity - self.equity) >= self.max_daily_loss:
                    self.log(f"Max daily loss exceeded: initial {self.initial_equity}, current {self.equity}")
                    self.running = False
                    return {"status": "stopped_max_daily_loss"}

                return {"status": "ok", "action": action, "signal": signal, "exec_out": out}
            except Exception as e:
                self.log(f"Tick failed: {e}")
                return {"status": "error", "error": str(e)}

    # ---- control ----
    def start(self):
        self.running = True
        self.log("Bot started.")

    def stop(self):
        self.running = False
        self.log("Bot stopped.")

    def state_snapshot(self):
        return {
            "running": self.running,
            "position": self.position,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "last_fetch_ts": self.last_fetch_ts,
            "logs_tail": self.logs[-100:]
        }


# ---------------------------
# Streamlit UI / controls (render)
# ---------------------------
def render():
    st.header("Trading Live (signals & execution) - Modo continuo (no bloqueante)")

    # ------------ Config UI ------------
    st.subheader("Execution & Risk configuration")
    col1, col2 = st.columns([2, 2])
    with col1:
        mode = st.selectbox("Mode", ["Paper/Testnet (sandbox)", "Mainnet"], index=0)
        exchange_id = st.text_input("Exchange id (ccxt)", value="binance")
        symbol = st.text_input("Symbol", value="BTC/USDT")
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=3)
        api_key = st.text_input("API Key", value="", type="password")
        api_secret = st.text_input("API Secret", value="", type="password")
    with col2:
        paper_mode = mode.startswith("Paper")
        max_pos_pct = st.number_input("Max position size (% equity)", min_value=0.001, max_value=100.0, value=2.0, step=0.1)
        max_daily_loss = st.number_input("Max daily loss (absolute)", value=100.0, step=1.0)
        max_orders_per_min = st.number_input("Max orders per minute", min_value=1, max_value=120, value=6)
        slippage_tolerance = st.number_input("Slippage tolerance (%)", min_value=0.0, max_value=5.0, value=0.5)
        tick_rounding = st.checkbox("Auto-quantize price/amount by market tick/lot", value=True)

    st.markdown("---")

    # ------------ Fetch / Buffer ------------
    st.subheader("Data / history")
    if st.button("Fetch latest OHLCV via CCXT"):
        try:
            ex = get_exchange(exchange_id, api_key, api_secret, sandbox=paper_mode)
            df = None
            try:
                from utils import fetch_ccxt_ohlcv
                df = fetch_ccxt_ohlcv(symbol=symbol, timeframe=timeframe, limit=1000, exchange_id=exchange_id, api_key=api_key, api_secret=api_secret, sandbox=paper_mode)
            except Exception:
                ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000)
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            st.session_state["fetched_df"] = df
            st.success(f"Fetched {len(df)} rows and stored in session as fetched_df.")
        except Exception as e:
            st.error(f"Fetch failed: {e}")

    df_full = st.session_state.get("fetched_df_full") or st.session_state.get("fetched_df")
    if df_full is None:
        st.info("No historical data loaded in session. Use the 'Datos' tab or Fetch OHLCV here.")
    else:
        st.write("Preview (last rows):")
        st.dataframe(df_full.tail(10))

    # Build history buffer for model input
    cfg = st.session_state.get("config", {})
    seq_len = int(cfg.get("seq_len", 32))
    horizon = int(cfg.get("horizon", 10))
    warmup = int(st.session_state.get("warmup_count", max(0, seq_len)))
    max_history = max(seq_len + horizon + 5, warmup + seq_len + 5)
    history_df = create_history_buffer_from_df(df_full, max_history) if df_full is not None else pd.DataFrame()
    st.session_state["history_buffer"] = history_df

    # ------------ Model inference controls ------------
    st.subheader("Model inference (one-shot and continuous)")
    model_path = st.session_state.get("model", None)
    scaler_path = st.session_state.get("scaler_path", None)
    meta = st.session_state.get("meta", {})

    st.write("Loaded model:", model_path)
    st.write("Loaded scaler:", scaler_path)
    st.write("Meta:", meta)

    # thresholds for trading rule
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        enter_threshold = st.number_input("Enter threshold (predicted return >)", value=0.0005, format="%.6f")
        exit_threshold = st.number_input("Exit threshold (predicted return <)", value=-0.0005, format="%.6f")
    with tcol2:
        position_pct = st.number_input("Position size (% equity) for signal", min_value=0.001, max_value=100.0, value=1.0, step=0.1)
        check_spread = st.checkbox("Check spread before sending live orders", value=True)

    # ------------ One-shot signal (kept for convenience) ------------
    if st.button("Run signal once (predict & decide)"):
        if model_path is None or meta is None:
            st.error("Primero entrena o carga un modelo en la pestaña Entrenamiento.")
        elif history_df is None or len(history_df) < seq_len:
            st.error(f"No hay suficiente historia para crear secuencia (necesitas >= seq_len={seq_len}).")
        else:
            feature_cols = meta.get("feature_cols", [c for c in history_df.columns if c not in ["close", "high", "low", "volume"]])
            if len(feature_cols) == 0:
                st.warning("No feature columns in meta; using ['close'] as fallback.")
                feature_cols = ["close"]
            try:
                X = prepare_model_input(history_df, feature_cols, seq_len, scaler_path=scaler_path)
                pred = predict_with_model(model_path, X)
                st.write("Prediction:", pred)
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                pred = None

            if pred is not None:
                pred_ret = pred.get("pred_ret", 0.0)
                pred_vol = pred.get("pred_vol", 0.0)
                st.write(f"pred_ret={pred_ret:.6f}, pred_vol={pred_vol:.6f}")

                signal = None
                if pred_ret > enter_threshold:
                    signal = "buy"
                elif pred_ret < exit_threshold:
                    signal = "sell"
                else:
                    signal = "hold"
                st.info(f"Signal: {signal}")

    st.markdown("---")

    # ------------ Continuous bot controls ------------
    st.subheader("Continuous trading bot")
    # ensure we have feature cols & model path
    feature_cols = meta.get("feature_cols", [c for c in history_df.columns if c not in ["close", "high", "low", "volume"]])
    if len(feature_cols) == 0:
        feature_cols = ["close"]

    # Create or retrieve bot in session state
    if "trading_bot" not in st.session_state:
        # instantiate bot only if model is present; otherwise store placeholder
        if model_path:
            bot = TradingBot(
                model_path=model_path,
                scaler_path=scaler_path,
                feature_cols=feature_cols,
                seq_len=seq_len,
                horizon=horizon,
                exchange_id=exchange_id,
                symbol=symbol,
                timeframe=timeframe,
                api_key=api_key,
                api_secret=api_secret,
                paper=paper_mode,
                max_pos_pct=max_pos_pct,
                max_daily_loss=max_daily_loss,
                max_orders_per_min=max_orders_per_min,
                slippage_tolerance_pct=slippage_tolerance,
                tick_rounding=tick_rounding
            )
            st.session_state["trading_bot"] = bot
        else:
            st.session_state["trading_bot"] = None

    bot = st.session_state.get("trading_bot", None)
    bot_running = bot is not None and bot.running

    col_start, col_stop, col_step = st.columns([1,1,2])
    with col_start:
        if st.button("Start bot"):
            if bot is None:
                st.error("No model loaded or bot not initialized. Carga un modelo y vuelve a intentar.")
            else:
                bot.start()
                st.success("Bot started.")
    with col_stop:
        if st.button("Stop bot"):
            if bot:
                bot.stop()
                st.success("Bot stopped.")
    with col_step:
        if st.button("Force single tick now"):
            if bot:
                # Force tick now (single-shot)
                res = bot.tick(history_df, enter_threshold, exit_threshold, position_pct)
                st.write("Tick result:", res)

    # Each render, if bot.running, call bot.tick non-blocking check (Streamlit rerun will control cadence)
    if bot and bot.running:
        res = bot.tick(history_df, enter_threshold, exit_threshold, position_pct)
        st.write("Auto tick:", res)

    # show bot status & logs
    st.markdown("**Bot state**")
    if bot:
        st.write(bot.state_snapshot())
        st.markdown("Logs (recent):")
        for l in bot.logs[-30:]:
            st.text(l)
    else:
        st.info("Bot not initialized (need model loaded).")

    st.markdown("---")

    # ------------ Show ledger & export ------------
    st.subheader("Trades ledger")
    ledger = read_ledger()
    if len(ledger) == 0:
        st.info("No trades recorded yet.")
    else:
        st.dataframe(ledger.tail(200))
        if st.button("Export ledger CSV"):
            st.write("Ledger saved to", str(LEDGER_PATH))
            st.success("Export done.")

    st.markdown("**Notes**: Este modo continuo funciona usando `bot.tick()` por renderización de Streamlit (no threads persistentes). Para convertirlo en servicio que corra constantemente fuera de Streamlit, encapsula `TradingBot.run_loop()` en un proceso separado/daemon con control robusto y persistencia.")

