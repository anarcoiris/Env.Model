import streamlit as st
import importlib, time
try:
    import fiboevo as fibo
except Exception:
    fibo = importlib.import_module("fiboevo")

from utils import fetch_ccxt_ohlcv
import ccxt

def render():
    st.header("Trading Live (Testnet/Mainnet)")
    mode = st.selectbox("Mode", ["Paper/Testnet (sandbox)", "Mainnet"])
    exchange_id = st.text_input("Exchange id", value="binance")
    symbol = st.text_input("Symbol", value="BTC/USDT")
    timeframe = st.selectbox("Timeframe", ["1m","5m","15m","30m","1h","4h","1d"], index=4)
    api_key = st.text_input("API Key", value="", type="password")
    api_secret = st.text_input("API Secret", value="", type="password")

    st.markdown("**Note**: this UI issues direct CCXT calls. Be careful when using Mainnet. Use Testnet / sandbox when possible.")

    if st.button("Fetch latest OHLCV via CCXT"):
        try:
            df = fetch_ccxt_ohlcv(symbol=symbol, timeframe=timeframe, limit=500, exchange_id=exchange_id, api_key=api_key, api_secret=api_secret, sandbox=(mode.startswith("Paper")))
            st.session_state["fetched_df"] = df
            st.success("Fetched latest OHLCV and stored in session")
        except Exception as e:
            st.error(f"Fetch failed: {e}")
    st.subheader("Trading configuration")
    paper_mode = st.checkbox("Paper trading (simulate)", value=True)
    api_mode = "Sandbox" if paper_mode else "Mainnet"
    max_pos_pct = st.number_input("Max position size (% of equity)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
    max_daily_loss = st.number_input("Max daily loss (absolute)", value=100.0, step=1.0)
    max_orders_per_min = st.number_input("Max orders per minute", min_value=1, max_value=60, value=6)
    slippage_tolerance = st.number_input("Slippage tolerance (pct)", min_value=0.0, max_value=5.0, value=0.25)
    tick_rounding = st.checkbox("Auto quantize price/amount by tick", value=True)
    order_type_default = st.selectbox("Default order type", ["limit", "market", "post_only"])

    st.markdown("---")
    st.subheader("Manual Order (ccxt)")
    side = st.selectbox("Side", ["buy","sell"])
    order_type = st.selectbox("Type", ["limit","market"])
    amount = st.number_input("Amount (units)", value=0.001)
    price = st.number_input("Price (for limit)", value=0.0)
    if st.button("Place order via CCXT"):
        try:
            exchange_cls = getattr(ccxt, exchange_id)
            exchange = exchange_cls({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
            if mode.startswith("Paper"):
                try:
                    exchange.set_sandbox_mode(True)
                except Exception:
                    pass
            params = {}
            if order_type == "market":
                order = exchange.create_market_order(symbol, side, amount, params=params)
            else:
                order = exchange.create_limit_order(symbol, side, amount, price, params=params)
            st.write("Order response:")
            st.json(order)
        except Exception as e:
            st.error(f"Order failed: {e}")
