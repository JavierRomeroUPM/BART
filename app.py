import streamlit as st
import pandas as pd
import numpy as np
import arviz as az
import pymc_bart as pmb

# 1. INICIALIZACIÓN DEL ESTADO
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# 2. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Simulador Ph BART - Doctorado", layout="wide")

@st.cache_resource
def load_engine():
    return az.from_netcdf("modelo_bart_final.nc")

try:
    idata = load_engine()
except Exception as e:
    st.error(f"❌ Error al cargar el motor: {e}")
    st.stop()

# 3. INTERFAZ DE USUARIO
st.title("🚀 Predictor Ph - Motor Bayesiano BART")
st.markdown("Inferencia de alta fidelidad. Las 8 variables se registran en el historial inferior.")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas")
        mo = st.number_input("Parámetro mo", 5.0, 32.0, 20.0, step=0.1)
        ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0, step=0.1)
        gsi = st.number_input("GSI", 10.0, 85.0, 50.0, step=0.1)
        
    with col2:
        st.subheader("⚙️ Variables No Analíticas")
        b = st.number_input("Ancho B (m)", 4.5, 22.0, 11.0, step=0.1)
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"])
        v_dil = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)
        v_for = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=1)
        v_rug = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=0)

    submit = st.form_submit_button("🎯 CALCULAR PREDICCIÓN", use_container_width=True)

# 4. LÓGICA DE CÁLCULO
if submit:
    # Mapeo numérico
    pp_val = 1.0 if v_pp == "Con Peso" else 0.0
    dil_val = 1.0 if v_dil == "Asociada" else 0.0
    for_val = 1.0 if v_for == "Axisimétrica" else 0.0
    rug_val = 1.0 if v_rug == "Rugoso" else 0.0
    
    # ORDEN DEL EXCEL (IMPORTANTE): mo, B, UCS, GSI, PP, Dil, Form, Rug
    vec = [mo, b, ucs, gsi, pp_val, dil_val, for_val, rug_val]
    
    with st.spinner("Calculando inferencia estable..."):
        # Acceso a muestras de la posterior
        mu_samples = idata.posterior["mu"].values.flatten()
        
        # Predicción central (Mediana)
        log_median = np.median(mu_samples)
        ph_resultado = np.expm1(log_median)
        
        # Incertidumbre Científica (Error estándar de la media)
        std_error_log = np.std(mu_samples) / np.sqrt(len(mu_samples))
        low_p = log_median - (1.96 * std_error_log)
        high_p = log_median + (1.96 * std_error_log)
        
        hdi_low = np.expm1(low_p)
        hdi_high = np.expm1(high_p)
        error_barra = (hdi_high - hdi_low) / 2

    # --- PRESENTACIÓN ---
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.success(f"### Ph Predicho: **{ph_resultado:.4f} MPa**")
        st.write(f"**Intervalo de Confianza (95%):** [{hdi_low:.4f} - {hdi_high:.4f}] MPa")
    
    with res_col2:
        st.metric("Incertidumbre (±)", f"{error_barra:.4f} MPa")

    # Guardar en historial con las 10 columnas (8 entradas + 2 resultados)
    nuevo_registro = {
        "mo": mo, 
        "B (m)": b, 
        "UCS (MPa)": ucs, 
        "GSI": gsi,
        "Peso P.": v_pp, 
        "Dilatancia": v_dil, 
        "Forma": v_for, 
        "Rugosidad": v_rug,
        "Ph Predicho": round(ph_resultado, 4),
        "Incertidumbre": round(error_barra, 4)
    }
    st.session_state["historial"].insert(0, nuevo_registro)

# 5. HISTORIAL COMPLETO
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial Detallado de Simulaciones")
    df_hist = pd.DataFrame(st.session_state["historial"])
    st.dataframe(df_hist, use_container_width=True, hide_index=True)
    
    # Opción para descargar el historial (Útil para tu tesis)
    csv = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Historial (CSV)", data=csv, file_name="simulaciones_bart.csv", mime="text/csv")
    
    if st.button("🗑️ Borrar Historial"):
        st.session_state["historial"] = []
        st.rerun()
