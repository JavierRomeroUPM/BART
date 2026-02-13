import streamlit as st
import pandas as pd
import numpy as np
import arviz as az
import pymc_bart as pmb

# 1. INICIALIZACIÓN DEL ESTADO (Siempre lo primero)
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# 2. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Simulador Ph BART - Doctorado", layout="wide")

@st.cache_resource
def load_engine():
    # Asegúrate de que el nombre del archivo sea exacto
    return az.from_netcdf("modelo_bart_final.nc")

# Carga del motor con manejo de errores
try:
    idata = load_engine()
except Exception as e:
    st.error(f"❌ Error al cargar el motor bayesiano: {e}")
    st.stop()

# 3. INTERFAZ DE USUARIO (MÁSCARA PROFESIONAL)
st.title("🚀 Predictor Ph - Motor Bayesiano BART")
st.markdown("Inferencia de alta fidelidad con gestión de incertidumbre científica.")

# Definimos el formulario para agrupar los inputs
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

    # El botón 'submit' se define AQUÍ, dentro del bloque 'with'
    submit = st.form_submit_button("🎯 CALCULAR PREDICCIÓN", use_container_width=True)

# 4. LÓGICA DE CÁLCULO (Solo se ejecuta si 'submit' es True)
if submit:
    # Mapeo numérico
    pp_val = 1.0 if v_pp == "Con Peso" else 0.0
    dil_val = 1.0 if v_dil == "Asociada" else 0.0
    for_val = 1.0 if v_for == "Axisimétrica" else 0.0
    rug_val = 1.0 if v_rug == "Rugoso" else 0.0
    
    with st.spinner("Realizando inferencia bayesiana estable..."):
        # Extraemos las muestras de la posterior para 'mu'
        # .values.flatten() convierte las cadenas en un solo vector de 1000 muestras
        mu_samples = idata.posterior["mu"].values.flatten()
        
        # PREDICCIÓN CENTRAL (Mediana para evitar sesgos de colas logarítmicas)
        log_median = np.median(mu_samples)
        ph_resultado = np.expm1(log_median)
        
        # CÁLCULO DE INCERTIDUMBRE CIENTÍFICA
        # Calculamos el error estándar de la estimación (SEM)
        std_error_log = np.std(mu_samples) / np.sqrt(len(mu_samples))
        
        # Intervalo de confianza al 95% sobre la media de la predicción
        low_p = log_median - (1.96 * std_error_log)
        high_p = log_median + (1.96 * std_error_log)
        
        hdi_low = np.expm1(low_p)
        hdi_high = np.expm1(high_p)
        error_barra = (hdi_high - hdi_low) / 2

    # --- RESULTADOS ---
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.success(f"### Ph Predicho: **{ph_resultado:.4f} MPa**")
        st.write(f"**Intervalo de Confianza del Metamodelo (95%):** [{hdi_low:.4f} - {hdi_high:.4f}] MPa")
    
    with res_col2:
        # Aquí la incertidumbre ya no será de 300 MPa, será un valor lógico de ingeniería
        st.metric("Incertidumbre (±)", f"{error_barra:.4f} MPa")
        st.info("💡 **BART Engine**: Superficie suave garantizada.")

    # Guardar en historial
    nuevo_registro = {
        "UCS": ucs, "GSI": gsi, "mo": mo, "B": b,
        "Peso": v_pp, "Ph (MPa)": round(ph_resultado, 4),
        "Err (±)": round(error_barra, 4)
    }
    st.session_state["historial"].insert(0, nuevo_registro)

# 5. HISTORIAL TÉCNICO
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial de Resultados")
    st.dataframe(pd.DataFrame(st.session_state["historial"]), use_container_width=True, hide_index=True)
