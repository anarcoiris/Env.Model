import streamlit as st
import importlib, torch, pandas as pd
try:
    import fiboevo as fibo
except Exception:
    fibo = importlib.import_module("fiboevo")

def render():
    st.header("Backtest")
    if st.session_state.get("model") is None:
        st.warning("No hay modelo entrenado en sesión. Entrena primero en la pestaña Entrenamiento.")
    model_path = st.session_state.get("model")
    meta = st.session_state.get("meta")
    if st.button("Run quick backtest (last segment)"):
        if st.session_state.get("fetched_df") is None:
            st.error("Datos faltantes. Carga o genera datos primero.")
            return
        df = st.session_state["fetched_df"].dropna().reset_index(drop=True)
        feature_cols = meta["feature_cols"] if meta else [c for c in df.columns if c not in ["close","high","low","volume"]]
        try:
            import joblib
            scaler = joblib.load("scaler.pkl")
            df[feature_cols] = scaler.transform(df[feature_cols])
        except Exception:
            st.info("No scaler found or failed to load. Using raw scaled features in session")
        df_test = df.iloc[-2000:].reset_index(drop=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = fibo.load_model(model_path, input_size=len(feature_cols), hidden=meta.get("hidden",64), device=device)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            return
        summary, trades = fibo.backtest_market_maker(df_test, model, feature_cols, seq_len=int(meta.get("seq_len",32)), horizon=int(meta.get("horizon",10)))
        st.write("Backtest summary")
        st.json(summary)
        if trades:
            df_tr = pd.DataFrame(trades)
            st.dataframe(df_tr.head(200))
            if st.button("Export trades to CSV"):
                df_tr.to_csv("trades_export.csv", index=False)
                st.success("Saved trades_export.csv")


# Simulation parameters (slippage & fees)
st.markdown("### Simulation parameters (slippage & fees)")
slippage_pct = st.number_input("Slippage per order (%)", value=0.1, format="%.4f")
fee_pct = st.number_input("Fee per trade (%)", value=0.04, format="%.4f")
if st.button("Run backtest with simulated execution (slippage & fees)"):
    if st.session_state.get("fetched_df") is None:
        st.error("No data in session")
    else:
        df_bt = st.session_state.get("fetched_df").dropna().reset_index(drop=True)
        meta = st.session_state.get("meta", {})
        feature_cols = meta.get("feature_cols", [c for c in df_bt.columns if c not in ['close','high','low','volume']])
        model_path = st.session_state.get("model")
        scaler_path = st.session_state.get("scaler_path")
        seq_len = int(st.session_state.get('config', {}).get('seq_len', 32))
        trades = []
        for i in range(seq_len, len(df_bt)-1):
            seg = df_bt.iloc[i-seq_len:i]
            try:
                X = prepare_model_input(seg, feature_cols, seq_len, scaler_path=scaler_path)
                pred = predict_with_model(model_path, X)
            except Exception:
                continue
            pred_ret = pred.get('pred_ret', 0.0)
            if pred_ret > 0.001:
                price = float(df_bt.iloc[i+1]['open'])
                slippage = price * (slippage_pct/100.0)
                exec_price = price + slippage
                fee = exec_price * (fee_pct/100.0)
                trades.append({'idx': i+1, 'side': 'buy', 'price': exec_price, 'fee': fee})
        st.write(f"Simulated {len(trades)} trades (sample 50):")
        st.dataframe(trades[:50])
