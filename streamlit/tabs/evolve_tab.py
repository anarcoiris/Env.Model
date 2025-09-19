import streamlit as st
import importlib, torch
try:
    import fiboevo as fibo
except Exception:
    fibo = importlib.import_module("fiboevo")

def render():
    st.header("Evolución (DEAP)")
    if not getattr(fibo, "_HAS_DEAP", False):
        st.warning("DEAP no está instalado. Instálalo con `pip install deap` para usar evolución.")
        return
    pop = st.number_input("Población", min_value=4, max_value=200, value=20)
    gens = st.number_input("Generaciones", min_value=1, max_value=200, value=10)
    if st.button("Run Evolution"):
        if st.session_state.get("fetched_df") is None:
            st.error("Carga datos primero.")
            return
        df = st.session_state["fetched_df"].dropna().reset_index(drop=True)
        feature_cols = [c for c in df.columns if c not in ["close","high","low","volume"]]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with st.spinner("Running evolution..."):
            pop_res, hof = fibo.run_evolution(df, feature_cols, pop_size=int(pop), gens=int(gens), device=device, seed=42, horizon=int(st.session_state['config'].get('horizon',10)))
        st.success("Evolution finished")
        st.write("Hall of Fame best individual:", hof[0])
        st.write("Sample population (first 5):", pop_res[:5])