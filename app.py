import streamlit as st
import pandas as pd
import numpy as np
import arviz as az
import pymc_bart as pmb

# Configuración profesional
st.set_page_config(page_title="Simulador Ph BART - Doctorado", layout="wide")

@st.cache_resource
def load_engine():
    return az.from_netcdf("modelo_bart_final.nc")

idata = load_engine()

st.title("🚀 Predictor Ph - Motor Bayesiano de Alta Fidelidad")
st.markdown("Inferencia BART: Transiciones suaves y gestión de incertidumbre científica.")

# --- FORMULARIO CON TU MÁSCARA ---
with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas")
        mo = st.number_input("Parámetro mo", 5.0, 32.0, 20.0, step=0.1)
        b = st.number_input("Ancho B (m)", 4.5, 22.0, 11.0, step=0.1)
        ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0, step=0.1)
        gsi = st.number_input("GSI", 10.0, 85.0, 50.0, step=0.1)
        
    with col2:
        st.subheader("⚙️ Variables No Analíticas")
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"])
        v_dil = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)
        v_for = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=1)
        v_rug = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=0)

    submit = st.form_submit_button("🎯 CALCULAR PREDICCIÓN", use_container_width=True)

if submit:
    # Mapeo numérico
    pp_val = 1 if v_pp == "Con Peso" else 0
    dil_val = 1 if v_dil == "Asociada" else 0
    for_val = 1 if v_for == "Axisimétrica" else 0
    rug_val = 1 if v_rug == "Rugoso" else 0
    
    # VECTOR DE ENTRADA (ORDEN IDÉNTICO A TU IMAGEN)
    # 0:mo, 1:B, 2:UCS, 3:GSI, 4:PP, 5:Dil, 6:Form, 7:Rug
    vec = [mo, b, ucs, gsi, pp_val, dil_val, for_val, rug_val]
    
    with st.spinner("Consultando motor bayesiano..."):
        mu_samples = idata.posterior["mu"]
        
        # Usamos la mediana para una predicción robusta
        ph_log_pred = np.median(mu_samples)
        ph_resultado = np.expm1(ph_log_pred)
        
        # Incertidumbre: Rango intercuartílico (más estable en geotecnia)
        low_p = np.percentile(mu_samples, 25)
        high_p = np.percentile(mu_samples, 75)
        hdi_low = np.expm1(low_p)
        hdi_high = np.expm1(high_p)
        error_barra = (hdi_high - hdi_low) / 2

    # --- RESULTADOS ---
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.success(f"### Ph Predicho: **{ph_resultado:.4f} MPa**")
        st.write(f"**Intervalo de Confianza (BART):** [{hdi_low:.2f} - {hdi_high:.2f}] MPa")
    
    with res_col2:
        st.metric("Incertidumbre", f"± {error_barra:.4f} MPa")
        st.info("💡 Transición suave garantizada.")

    # Historial
    if "historial" not in st.session_state: st.session_state["historial"] = []
    st.session_state["historial"].insert(0, {"mo": mo, "B": b, "UCS": ucs, "GSI": gsi, "Ph": round(ph_resultado, 4)})

if st.session_state["historial"]:
    st.table(pd.DataFrame(st.session_state["historial"]))
