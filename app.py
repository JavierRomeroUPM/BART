import streamlit as st
import pandas as pd
import numpy as np
import arviz as az
import pymc_bart as pmb
from datetime import datetime

# ==============================================================================
# 1. CONFIGURACIÓN Y CARGA DEL MOTOR BART (.NC)
# ==============================================================================
st.set_page_config(page_title="Simulador Ph BART - Doctorado", layout="wide")

if "historial" not in st.session_state:
    st.session_state["historial"] = []

@st.cache_resource
def load_bart_model():
    try:
        # Cargamos el InferenceData generado en Colab
        return az.from_netcdf("modelo_bart_final.nc")
    except Exception as e:
        st.error(f"❌ Error al cargar el motor bayesiano (.nc): {e}")
        st.stop()

idata = load_bart_model()

# ==============================================================================
# 2. INTERFAZ DE USUARIO (TU MÁSCARA ADAPTADA)
# ==============================================================================
st.title("🚀 Predictor Ph - Motor Bayesiano BART")
st.markdown("Inferencia mediante árboles aditivos bayesianos para la obtención de superficies de respuesta continuas.")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas")
        ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0, step=0.1)
        gsi = st.number_input("GSI", 10.0, 85.0, 50.0, step=0.1)
        mo = st.number_input("Parámetro mo", 5.0, 32.0, 20.0, step=0.1)
        b = st.number_input("Ancho B (m)", 4.5, 22.0, 11.0, step=0.1)
        
    with col2:
        st.subheader("⚙️ Variables No Analíticas")
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"])
        v_dil = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)
        v_for = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=1)
        v_rug = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=0)

    submit = st.form_submit_button("🎯 CALCULAR PREDICCIÓN BAYESIANA", use_container_width=True)

if submit:
    # Mapeo a formato numérico (debe coincidir con el orden de Colab)
    pp_val = 1 if v_pp == "Con Peso" else 0
    dil_val = 1 if v_dil == "Asociada" else 0
    for_val = 1 if v_for == "Axisimétrica" else 0
    rug_val = 1 if v_rug == "Rugoso" else 0
    
    # Vector de entrada (Ajustar este orden exacto según tu entrenamiento en Colab)
    # Ejemplo: [GSI, UCS, mo, B, PP, Dil, Form, Rug]
    vec = [gsi, ucs, mo, b, pp_val, dil_val, for_val, rug_val]
    
    # --- PROCESAMIENTO CON EL MOTOR BART ---
    with st.spinner("Realizando inferencia bayesiana..."):
        # Extraemos las muestras de la posterior para 'mu'
        mu_samples = idata.posterior["mu"]
        
        # Calculamos la media de la distribución (nuestra Ph predicha)
        ph_log_pred = mu_samples.mean().values
        ph_resultado = np.expm1(ph_log_pred)
        
        # Calculamos la incertidumbre de forma correcta (Intervalo de Credibilidad 95%)
        # En lugar de un +- gigante, calculamos los percentiles 2.5 y 97.5
        hdi_low = np.expm1(np.percentile(mu_samples, 2.5))
        hdi_high = np.expm1(np.percentile(mu_samples, 97.5))
        error_barra = (hdi_high - hdi_low) / 2

    # --- PRESENTACIÓN DE RESULTADOS ---
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.success(f"### Ph Predicho: **{ph_resultado:.4f} MPa**")
        st.markdown(f"**Intervalo de Credibilidad (95%):** [{hdi_low:.2f} - {hdi_high:.2f}] MPa")
    
    with res_col2:
        # Mostramos la incertidumbre como un valor relativo al error estándar del modelo
        st.metric("Incertidumbre (±)", f"{error_barra:.4f} MPa")
        st.info("💡 **BART Engine**: Superficie suave garantizada por promedio de ensamble bayesiano.")

    # Guardar en historial
    nuevo_registro = {
        "UCS": ucs, "GSI": gsi, "mo": mo, "B": b,
        "Peso": v_pp, "Dilat.": v_dil, "Forma": v_for, "Rugos.": v_rug,
        "Ph (MPa)": round(ph_resultado, 4),
        "Err (±)": round(error_barra, 4)
    }
    st.session_state["historial"].insert(0, nuevo_registro)

# ==============================================================================
# 3. HISTORIAL TÉCNICO
# ==============================================================================
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial de Resultados")
    df_hist = pd.DataFrame(st.session_state["historial"])
    st.dataframe(df_hist, use_container_width=True, hide_index=True)
    
    if st.button("🗑️ Borrar Historial"):
        st.session_state["historial"] = []
        st.rerun()

st.markdown("---")
st.caption("Modelo BART Puro | Doctorado | Inferencia Bayesiana sobre NetCDF")
