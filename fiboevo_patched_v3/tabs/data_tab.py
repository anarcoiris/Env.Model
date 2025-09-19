import streamlit as st
import pandas as pd
from pathlib import Path
import importlib
# import fiboevo from the same directory as the app; attempt direct import
try:
    import fiboevo as fibo
except Exception:
    fibo = importlib.import_module("fiboevo")

from utils import fetch_ccxt_ohlcv, save_ohlcv_json, load_ohlcv_json

DATA_JSON = Path("fetched_ohlcv.json")

# utils (o al inicio de tabs/data_tab.py)
def compute_warmup_count(seq_len: int, window_long: int = 50, extra_lookback: int = 200) -> int:
    """
    Número conservador de filas necesarias para "calentar" indicadores.
    Ajusta aquí si tus indicadores necesitan más.
    """
    candidates = [int(seq_len), int(window_long), int(extra_lookback), 200, 30, 50]
    return max(candidates)

def display_warmup_info(st, df_len: int, warmup: int):
    st.info(f"La creación de indicadores requiere al menos {warmup} barras de historial.\n"
            f"Filas originales: {df_len}. Filas disponibles para entrenamiento: {max(0, df_len - warmup)}")

def render():
    st.header("Datos")
    source = st.selectbox("Fuente", ["CCXT (exchange)", "InfluxDB (opcional)", "Simulación OU (fallback)"])

    if source == "CCXT (exchange)":
        exchange = st.text_input("Exchange id (ccxt)", value="binance")
        symbol = st.text_input("Symbol", value="BTC/USDT")
        timeframe = st.selectbox("Timeframe", ["1m","5m","15m","30m","1h","4h","1d"], index=4)
        limit = st.number_input("Limit (candles)", min_value=50, max_value=100000, value=1000, step=50)
        if st.button("Fetch OHLCV from exchange (ccxt)"):
            with st.spinner("Fetching..."):
                try:
                    df = fetch_ccxt_ohlcv(symbol=symbol, timeframe=timeframe, limit=int(limit), exchange_id=exchange)
                    st.session_state["fetched_df"] = df
                    save_ohlcv_json(df, DATA_JSON)
                    st.success(f"Fetched {len(df)} rows and saved to {DATA_JSON}")
                except Exception as e:
                    st.error(f"Fetch failed: {e}")

    elif source == "InfluxDB (opcional)":
        exchange = st.text_input("Exchange id ", value="binance")
        symbol = st.text_input("Symbol", value="BTC/USDT")
        timeframe = st.selectbox("Timeframe", ["1m","5m","15m","30m","1h","4h","1d"], index=4)
        limit = st.number_input("Limit (candles)", min_value=50, max_value=100000, value=1000, step=50)
        st.info("Si tiene InfluxDB y el cliente instalado, configure variables de entorno y use `fiboevo.load_data_from_influx`.")
        if st.button("Try load from Influx (fiboevo)"):
            with st.spinner("Querying Influx..."):
                try:
                    arr = fibo.load_data_from_influx()
                    noise = 0.001 * abs(pd.np.random.randn(len(arr)))
                    df = pd.DataFrame({"close": arr, "high": arr * (1+noise), "low": arr * (1-noise)})
                    st.session_state["fetched_df"] = df
                    save_ohlcv_json(df, DATA_JSON)
                    st.success("Loaded from Influx and stored locally")
                except Exception as e:
                    st.error(f"Influx load failed: {e}")

    else:
        st.info("OU simulator fallback (quick synthetic series)")
        steps = st.number_input("OU steps", min_value=100, max_value=20000, value=2000)
        seed = st.number_input("Seed", value=42)
        if st.button("Generate OU"):
            close, high, low = fibo.simulate_ou(steps=int(steps), seed=int(seed))
            df = pd.DataFrame({"close": close, "high": high, "low": low})
            st.session_state["fetched_df"] = df
            save_ohlcv_json(df, DATA_JSON)
            st.success("Synthetic data generated and saved")

    # preview and compute features
    if st.session_state.get("fetched_df") is not None:
        df = st.session_state["fetched_df"]
        st.subheader("Preview")
        st.dataframe(df.head(200))

        if st.button("Compute technical features (fiboevo.add_technical_features)"):
            try:
                df_feat = fibo.add_technical_features(df["close"].values, high=df.get("high").values if "high" in df else None, low=df.get("low").values if "low" in df else None, volume=df.get("volume").values if "volume" in df else None, window_long=50)
                st.session_state["fetched_df"] = df_feat
                save_ohlcv_json(df_feat, DATA_JSON)
                st.success("Features computed and stored in session.")
                st.dataframe(df_feat.head(50))
            except Exception as e:
                st.error(f"Feature engineering failed: {e}")

    st.markdown('---')
    if DATA_JSON.exists():
        st.write(f"Local JSON: {DATA_JSON} (size: {DATA_JSON.stat().st_size} bytes)")
