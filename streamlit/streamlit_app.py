import streamlit as st
import importlib
import numpy as np
import pandas as pd
pd.np = np

st.set_page_config(page_title="FiboEvo — Control Panel", layout="wide")

st.title("FiboEvo — Control Panel (Streamlit Modular)")
st.markdown("Interfaz modular para controlar datos, entrenamiento, evolución, backtest y trading (testnet/mainnet).")

# persistent session state
if "fetched_df" not in st.session_state:
    st.session_state["fetched_df"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "scaler_path" not in st.session_state:
    st.session_state["scaler_path"] = None
if "meta" not in st.session_state:
    st.session_state["meta"] = None

tabs = ["📊 Datos", "⚙️ Configuración", "🤖 Entrenamiento", "🧬 Evolución", "📈 Backtest", "🚀 Trading Live", "📁 Artefactos"]
sel = st.sidebar.radio("Navegación", tabs)

# Global config store in session_state for tabs to share
if "config" not in st.session_state:
    st.session_state["config"] = {
        "seq_len": 32,
        "epochs": 20,
        "hidden": 64,
        "lr": 1e-3,
        "horizon": 10,
        "do_evolve": False,
        "evolve_pop": 20,
        "evolve_gens": 10,
    }

def _import_and_render(module_name: str, fn: str = "render"):
    try:
        mod = importlib.import_module(f"tabs.{module_name}")
        render_fn = getattr(mod, fn)
        render_fn()
    except Exception as e:
        st.error(f"Could not import or render module tabs.{module_name}: {e}")

if sel == "📊 Datos":
    _import_and_render("data_tab")
elif sel == "⚙️ Configuración":
    _import_and_render("utils_tab", "render_config")
elif sel == "🤖 Entrenamiento":
    _import_and_render("train_tab")
elif sel == "🧬 Evolución":
    _import_and_render("evolve_tab")
elif sel == "📈 Backtest":
    _import_and_render("backtest_tab")
elif sel == "🚀 Trading Live":
    _import_and_render("trading_tab")
elif sel == "📁 Artefactos":
    _import_and_render("utils_tab", "render_artifacts")
else:
    st.write("Seleccione una pestaña.")