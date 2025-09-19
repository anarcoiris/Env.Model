import streamlit as st
import torch, json
from pathlib import Path
import importlib
try:
    import fiboevo as fibo
except Exception:
    fibo = importlib.import_module("fiboevo")

MODEL_PATH = Path("streamlit_model.pt")
META_PATH = Path("streamlit_meta.json")
SCALER_PATH = Path("scaler.pkl")

def render():
    st.header("Entrenamiento")
    cfg = st.session_state.get("config", {})
    seq_len = st.slider("seq_len", 8, 128, value=cfg.get("seq_len", 32))
    epochs = st.slider("epochs", 1, 100, value=cfg.get("epochs", 20))
    hidden = st.slider("hidden", 16, 512, value=cfg.get("hidden", 64))
    lr = st.number_input("learning rate", value=float(cfg.get("lr", 1e-3)), format="%.6f")
    horizon = st.number_input("horizon", min_value=1, value=cfg.get("horizon", 10))

    st.session_state["config"].update({"seq_len": seq_len, "epochs": epochs, "hidden": hidden, "lr": lr, "horizon": horizon})

    if st.button("Train model (final)"):
        if st.session_state.get("fetched_df") is None:
            st.error("No hay datos en sesión. Carga datos en la pestaña Datos.")
            return
        df = st.session_state["fetched_df"]
        st.info("Preparing dataset and scaler...")
        feature_cols = [c for c in df.columns if c not in ["close", "high", "low", "volume"]]
        df_proc = df.dropna().reset_index(drop=True)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        try:
            df_proc[feature_cols] = scaler.fit_transform(df_proc[feature_cols])
        except Exception as e:
            st.warning(f"Scaling failed: {e}")
        try:
            X, y_ret, y_vol = fibo.create_sequences_from_df(df_proc, feature_cols, seq_len=seq_len, horizon=horizon)
        except Exception as e:
            st.error(f"Could not create sequences: {e}")
            return
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_ret), torch.from_numpy(y_vol))
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = fibo.LSTM2Head(input_size=X.shape[2], hidden_size=hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        st.info("Training... this may take a while depending on your machine.")
        progress = st.progress(0)
        for e in range(epochs):
            loss = fibo.train_epoch(model, loader, opt, device, alpha_vol_loss=0.5)
            progress.progress((e+1)/epochs)
            st.write(f"Epoch {e+1}/{epochs} loss={loss:.6f}")
        fibo.save_model(model, MODEL_PATH, {"feature_cols": feature_cols, "seq_len": seq_len, "horizon": horizon, "hidden": hidden}, META_PATH)
        try:
            import joblib
            joblib.dump(scaler, SCALER_PATH)
            st.session_state["scaler_path"] = str(SCALER_PATH)
        except Exception:
            st.warning("Could not save scaler (joblib missing).")
        st.success(f"Training finished. Model saved to {MODEL_PATH}")
        st.session_state["model"] = str(MODEL_PATH)
        st.session_state["meta"] = {"feature_cols": feature_cols, "seq_len": seq_len, "horizon": horizon, "hidden": hidden}