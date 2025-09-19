# tabs/utils_tab.py
import streamlit as st
from pathlib import Path
import json

def render_config():
    st.header("Configuración Global")
    cfg = st.session_state.get("config", {})
    seq_len = st.number_input("seq_len", value=int(cfg.get("seq_len",32)))
    epochs = st.number_input("epochs", value=int(cfg.get("epochs",20)))
    hidden = st.number_input("hidden", value=int(cfg.get("hidden",64)))
    lr = st.number_input("lr", value=float(cfg.get("lr",1e-3)))
    horizon = st.number_input("horizon", value=int(cfg.get("horizon",10)))
    do_evolve = st.checkbox("Activar evolución (DEAP)", value=bool(cfg.get("do_evolve", False)))
    st.session_state["config"].update({
        "seq_len": int(seq_len),
        "epochs": int(epochs),
        "hidden": int(hidden),
        "lr": float(lr),
        "horizon": int(horizon),
        "do_evolve": do_evolve
    })
    st.success("Config guardada en sesión.")

def render_artifacts():
    st.header("Artefactos guardados")
    p_model = Path("streamlit_model.pt")
    p_meta = Path("streamlit_meta.json")
    p_scaler = Path("scaler.pkl")
    files = []
    for p in [p_model, p_meta, p_scaler, Path("trades_export.csv"), Path("fetched_ohlcv.json")]:
        files.append({"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() else None})
    st.table(files)
    if st.button("List files in working dir"):
        import os
        st.write(os.listdir("."))
