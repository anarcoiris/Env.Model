
import streamlit as st
import torch, json, math
import pandas as pd
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
    seq_len = st.session_state["config"].get("seq_len", 32)
    epochs = st.session_state["config"].get("epochs", 20)
    hidden = st.session_state["config"].get("hidden", 64)
    lr = float(st.session_state["config"].get("lr", 1e-3))
    horizon = int(st.session_state["config"].get("horizon", 10))

    st.session_state["config"].update({"seq_len": seq_len, "epochs": epochs, "hidden": hidden, "lr": lr, "horizon": horizon})

    if st.button("Train model (final)"):
        if st.session_state.get("fetched_df") is None:
            st.error("No hay datos en sesión. Carga datos en la pestaña Datos.")
            return
        df = st.session_state["fetched_df"]
        st.info("Preparing dataset and scaler...")

        # Determine feature columns: all columns except base price cols
        feature_cols = [c for c in df.columns if c not in ["close", "high", "low", "volume"]]
        # fallback if none found
        if len(feature_cols) == 0:
            st.warning("No feature columns detected (only base price columns present). Using 'close' as fallback feature. Consider computing technical features in the Data tab before training.")
            feature_cols = ["close"]

        # Drop rows with NaNs and optionally trim warmup_count
        warmup = st.session_state.get("warmup_count", 0)
        df_proc = df.dropna().reset_index(drop=True)
        if warmup and len(df_proc) > warmup:
            st.info(f"Trimming first {warmup} rows (warm-up) before creating sequences for training")
            df_proc = df_proc.iloc[warmup:].reset_index(drop=True)

        # Scaling features robustly
        try:
            from sklearn.preprocessing import StandardScaler
        except Exception:
            st.error("scikit-learn is required (sklearn). Install with 'pip install scikit-learn'")
            return

        scaler = StandardScaler()
        try:
            X_to_scale = df_proc[feature_cols]
            if X_to_scale.shape[1] == 0:
                raise ValueError("No feature columns to scale (zero width)")
            df_proc.loc[:, feature_cols] = scaler.fit_transform(X_to_scale)
        except Exception as e:
            st.warning(f"Scaling failed: {e}. Attempting fallback coercion to numeric and retry.")
            try:
                df_proc[feature_cols] = df_proc[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
                df_proc.loc[:, feature_cols] = scaler.fit_transform(df_proc[feature_cols])
            except Exception as e2:
                st.error(f"Scaling absolutely failed: {e2}")
                return

        # Create sequences
        try:
            X, y_ret, y_vol = fibo.create_sequences_from_df(df_proc, feature_cols, seq_len=seq_len, horizon=horizon)
        except Exception as e:
            st.error(f"Could not create sequences: {e}")
            return

        # Build dataloader
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        try:
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_ret), torch.from_numpy(y_vol))
        except Exception as e:
            st.error(f"Failed creating TensorDataset: {e}")
            return
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # sanity checks to avoid cuDNN errors
        input_size = X.shape[2] if X.ndim == 3 else (X.shape[1] if X.ndim == 2 else 0)
        if input_size == 0:
            st.error("Computed input_size is zero. Check feature columns and that X has the expected shape. Aborting training.")
            return

        try:
            model = fibo.LSTM2Head(input_size=input_size, hidden_size=hidden).to(device)
        except Exception as e_model:
            st.warning(f"Model allocation/transfer to device failed: {e_model}. Falling back to CPU.")
            device = torch.device('cpu')
            model = fibo.LSTM2Head(input_size=input_size, hidden_size=hidden).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        st.info("Training... this may take a while depending on your machine.")
        progress = st.progress(0)
        for e in range(epochs):
            try:
                loss = fibo.train_epoch(model, loader, opt, device, alpha_vol_loss=0.5)
            except Exception as e_train:
                st.error(f"Training failed at epoch {e+1}: {e_train}")
                return
            progress.progress(int((e+1)/epochs*100))
            st.write(f"Epoch {e+1}/{epochs} loss={loss:.6f}")

        # save artifacts
        try:
            fibo.save_model(model, MODEL_PATH, {"feature_cols": feature_cols, "seq_len": seq_len, "horizon": horizon, "hidden": hidden}, META_PATH)
        except Exception as e:
            st.error(f"Failed to save model: {e}")
            return

        try:
            import joblib
            joblib.dump(scaler, SCALER_PATH)
            st.session_state["scaler_path"] = str(SCALER_PATH)
        except Exception:
            st.warning("Could not save scaler (joblib missing).")

        st.success(f"Training finished. Model saved to {MODEL_PATH}")
        st.session_state["model"] = str(MODEL_PATH)
        st.session_state["meta"] = {"feature_cols": feature_cols, "seq_len": seq_len, "horizon": horizon, "hidden": hidden}
