# tabs/trading_tab.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import time, math, json
import importlib

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

# Import fiboevo functions/models
try:
    import fiboevo as fibo
except Exception:
    fibo = importlib.import_module("fiboevo")

LEDGER_PATH = Path("trades_ledger.csv")


# ---------------------------
# Utility helpers
# ---------------------------
def get_exchange(exchange_id: str, api_key: str = None, api_secret: str = None, sandbox: bool = True):
    if ccxt is None:
        raise RuntimeError("ccxt no está instalado. Instala con `pip install ccxt` para usar exchanges.")
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

        # precision may be in m['precision'] or limits.stepSize
        tick = None
        lot = None
        if isinstance(m.get("precision"), dict):
            tick = m["precision"].get("price", None)
            lot = m["precision"].get("amount", None)
        # fallback: try limits->stepSize
        if tick is None and isinstance(m.get("limits"), dict):
            if isinstance(m["limits"].get("price"), dict):
                tick = m["limits"]["price"].get("min", None)
            if isinstance(m["limits"].get("amount"), dict):
                lot = m["limits"]["amount"].get("min", None)

        # Convert to rounded tick if possible
        if tick is not None and tick > 0:
            # sometimes precision is 8 meaning decimals; try to convert to tick value:
            if tick < 1 and tick > 0:
                return float(tick), float(lot) if lot else None
        return (float(tick) if tick else None, float(lot) if lot else None)
    except Exception:
        return None, None


def quantize_price(price: float, tick: float):
    if tick is None or tick == 0:
        return float(price)
    # floor to tick
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
    df = read_ledger()
    df = df.append(entry, ignore_index=True)
    df.to_csv(LEDGER_PATH, index=False)


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

    # select last seq_len rows
    seg = df_segment.iloc[-seq_len:].copy().reset_index(drop=True)
    X = seg[feature_cols].values.astype(np.float32)

    # apply scaler if provided
    if scaler_path and Path(scaler_path).exists() and joblib is not None:
        try:
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X)
        except Exception as e:
            st.warning(f"Fallo al cargar/transformar con scaler: {e} — se usará X sin escalar.")
    # shape to (1, seq_len, features)
    return np.expand_dims(X, axis=0)


def predict_with_model(model_path: str, X_numpy):
    """
    Carga modelo desde model_path usando fibo.load_model (si existe) o cargando torch directly.
    Devuelve una diccionario de outputs según fiboevo.predict API (supuesto: retorno & vol).
    """
    device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
    if not Path(model_path).exists():
        raise FileNotFoundError("Model path not found: " + str(model_path))

    # preferimos usar la función fibo.load_model si existe
    try:
        model = fibo.load_model(model_path, input_size=X_numpy.shape[2], device=device)
    except Exception:
        # fallback: load state_dict into LSTM2Head if available
        try:
            model = fibo.LSTM2Head(input_size=X_numpy.shape[2], hidden_size=64)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar el modelo con fibo.load_model ni fallback: {e}")

    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_numpy).to(device)
        out_ret, out_vol = model(xt)  # dependiente de la implementacion original
        # convert to numpy
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
# ---------------------------
def execute_order(exchange, symbol: str, side: str, amount: float, price: float, order_type: str = "limit",
                  paper: bool = True, slippage_tolerance_pct: float = 0.5, tick: float = None, lot: float = None):
    """
    Ejecuta la orden en modo paper (simulado) o real (ccxt exchange).
    Retorna un dic con info del fill.
    """
    now_ts = pd.Timestamp.utcnow().isoformat()
    if paper:
        # simulate fill with slippage: market orders fill at mid price; limit may fill depending on price vs last close
        executed_price = price
        # apply slippage proportional to tolerance
        slippage = price * (slippage_tolerance_pct / 100.0)
        # simulate worst-case slippage direction
        if side.lower() == "buy":
            executed_price = price + slippage
        else:
            executed_price = price - slippage
        if tick:
            executed_price = quantize_price(executed_price, tick)
        if lot:
            amount = quantize_amount(amount, lot)
        pnl = None
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
        # quantize
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
# Streamlit UI / controls
# ---------------------------
def render():
    st.header("Trading Live (signals & execution)")

    # ------------
    # Config UI
    # ------------
    st.subheader("Execution & Risk configuration")
    col1, col2 = st.columns([2, 2])
    with col1:
        mode = st.selectbox("Mode", ["Paper/Testnet (sandbox)", "Mainnet"], index=0)
        exchange_id = st.text_input("Exchange id (ccxt)", value="binance")
        symbol = st.text_input("Symbol", value="BTC/USDT")
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=5)
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

    # ------------
    # Fetch / Buffer
    # ------------
    st.subheader("Data / history")
    if st.button("Fetch latest OHLCV via CCXT"):
        try:
            ex = get_exchange(exchange_id, api_key, api_secret, sandbox=paper_mode)
            df = None
            # Use utils.fetch_ccxt_ohlcv if present in repo or fallback
            try:
                from utils import fetch_ccxt_ohlcv
                df = fetch_ccxt_ohlcv(symbol=symbol, timeframe=timeframe, limit=1000, exchange_id=exchange_id, api_key=api_key, api_secret=api_secret, sandbox=paper_mode)
            except Exception:
                # fallback direct call
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

    # ------------
    # Model inference controls
    # ------------
    st.subheader("Model inference (one-shot)")
    model_path = st.session_state.get("model", None)
    scaler_path = st.session_state.get("scaler_path", None)
    meta = st.session_state.get("meta", {})

    st.write("Loaded model:", model_path)
    st.write("Loaded scaler:", scaler_path)
    st.write("Meta:", meta)

    # threshold selectors for trading rule
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        enter_threshold = st.number_input("Enter threshold (predicted return >)", value=0.0005, format="%.6f")
        exit_threshold = st.number_input("Exit threshold (predicted return <)", value=-0.0005, format="%.6f")
    with tcol2:
        position_pct = st.number_input("Position size (% equity) for signal", min_value=0.001, max_value=100.0, value=1.0, step=0.1)
        check_spread = st.checkbox("Check spread before sending live orders", value=True)

    # ------------
    # Run a single signal + optionally execute
    # ------------
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

                # Basic rule: enter long if pred_ret > enter_threshold, enter short if pred_ret < exit_threshold
                signal = None
                if pred_ret > enter_threshold:
                    signal = "buy"
                elif pred_ret < exit_threshold:
                    signal = "sell"
                else:
                    signal = "hold"
                st.info(f"Signal: {signal}")

                # prepare execution parameters
                last_close = float(history_df.iloc[-1]["close"])
                # position sizing: percent of equity -> here we use a placeholder equity value (user must adapt)
                equity = 10000.0  # TODO: Hook real account balance via exchange.fetch_balance()
                size_usd = equity * (position_pct / 100.0)
                amount = size_usd / last_close
                # quantize price/amount using exchange precision if available
                tick, lot = (None, None)
                exchange = None
                if ccxt is not None:
                    try:
                        exchange = get_exchange(exchange_id, api_key, api_secret, sandbox=paper_mode)
                        m_tick, m_lot = get_market_precision(exchange, symbol)
                        tick, lot = (m_tick, m_lot)
                    except Exception as e:
                        st.warning(f"Could not get market precision: {e}")

                # compute order params
                order_type = "market"  # default (quicker); could be limit
                price = last_close
                if signal in ("buy", "sell"):
                    st.write("Prepared order:", {"side": signal, "amount": amount, "price": price, "tick": tick, "lot": lot})
                    # show execute controls
                    if st.button("Execute this order (single-shot)"):
                        # risk checks
                        if not paper_mode and check_spread:
                            try:
                                ob = exchange.fetch_order_book(symbol)
                                spread = ob['asks'][0][0] - ob['bids'][0][0]
                                spread_pct = spread / last_close * 100.0
                                st.write(f"Spread pct: {spread_pct:.4f}%")
                                if spread_pct > 1.0:
                                    st.error("Spread too high, aborting live order.")
                                    raise RuntimeError("High spread")
                            except Exception as e:
                                st.warning(f"Could not check spread: {e}")

                        try:
                            executed = execute_order(
                                exchange=exchange,
                                symbol=symbol,
                                side=signal,
                                amount=amount,
                                price=price,
                                order_type=order_type,
                                paper=paper_mode,
                                slippage_tolerance_pct=slippage_tolerance,
                                tick=tick if tick_rounding else None,
                                lot=lot if tick_rounding else None
                            )
                            st.success("Order executed (or simulated). See ledger below.")
                            st.json(executed)
                        except Exception as e:
                            st.error(f"Execution failed: {e}")
                else:
                    st.info("No action: hold signal.")

    # ------------
    # Show ledger & export
    # ------------
    st.markdown("---")
    st.subheader("Trades ledger")
    ledger = read_ledger()
    if len(ledger) == 0:
        st.info("No trades recorded yet.")
    else:
        st.dataframe(ledger.tail(200))
        if st.button("Export ledger CSV"):
            st.write("Ledger saved to", str(LEDGER_PATH))
            st.success("Export done.")

    st.markdown("**Notes**: This UI executes single-shot orders. If you want continuous automated trading you should implement scheduling/monitoring and robust retry/abort logic. Use Paper/Testnet first.")
