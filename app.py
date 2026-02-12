import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ==============================================================================
# 1. CARGA DE ACTIVOS Y FUNCIONES
# ==============================================================================
st.set_page_config(page_title="Simulador Ph Geotécnico - BART", layout="wide")

@st.cache_resource
def load_bart_model():
    try:
        # Asegúrate de que este nombre de archivo sea exacto al que subiste a GitHub
        with open("modelo_bart_final.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()

# Carga de datos del modelo
assets = load_bart_model()
modelos_ensemble = assets['modelo']
metodo = assets['metodo']
v_disc = assets['valores_disc']

def predict_with_uncertainty(x_input, model_list):
    # Predicción logarítmica de cada árbol del ensamble
    preds_log = np.array([m.predict(x_input) for m in model_list])
    m_log = np.mean(preds_log, axis=0)
    s_log = np.std(preds_log, axis=0)
    
    # Revertir logaritmo
    ph_mean = np.expm1(m_log)[0]
    # Intervalo de confianza (95%)
    low_ph = np.expm1(m_log - 1.96 * s_log)[0]
    high_ph = np.expm1(m_log + 1.96 * s_log)[0]
    
    return ph_mean, low_ph, high_ph, s_log[0]

# ==============================================================================
# 2. INTERFAZ DE USUARIO
# ==============================================================================
st.title("🎯 Predictor de Presión de Hundimiento (Ph)")
st.subheader("Metamodelo de Alta Fidelidad mediante BART-Ensemble")

# Inicialización del historial
if "hist" not in st.session_state: 
    st.session_state.hist = []

with st.form("input_form"):
    c1, c2 = st.columns(2)
    with c1:
        st.info("🧪 Parámetros del Macizo (Analíticos)")
        ucs = st.number_input("UCS (MPa)", 5.0, 110.0, 50.0, step=0.1)
        gsi = st.number_input("GSI", 5.0, 100.0, 50.0, step=0.1)
        mo = st.number_input("mo (Hoek-Brown)", 5.0, 32.0, 20.0, step=0.1)
    with c2:
        st.info("⚙️ Geometría y Condiciones (No Analíticos)")
        b = st.number_input("Ancho B (m)", 1.0, 40.0, 11.0, step=0.1)
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"])
        v_dil = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)
        v_for = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=1)
        v_rug = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=0)

    calculate = st.form_submit_button("EJECUTAR PREDICCIÓN", use_container_width=True)

if calculate:
    # Mapeo a vector numérico
    # mo, B, UCS, GSI, Peso, Dilat, Forma, Rugos
    vec = [[
        mo, 
        b, 
        ucs, 
        gsi, 
        1 if v_pp == "Con Peso" else 0, 
        1 if v_dil == "Asociada" else 0, 
        1 if v_for == "Axisimétrica" else 0, 
        1 if v_rug == "Rugoso" else 0
    ]]
    
    ph, low, high, sigma = predict_with_uncertainty(np.array(vec), modelos_ensemble)
    
    # ZONA DE RESULTADOS
    st.divider()
    res1, res2 = st.columns([2, 1])
    
    with res1:
        st.success(f"### Presión de Hundimiento Predicha: **{ph:.3f} MPa**")
        st.write(f"Intervalo de Confianza (95%): **[{low:.2f} - {high:.2f}] MPa**")
        
        # Alerta de Extrapolación basada en los rangos de entrenamiento reales
        # Ajustado a tus límites: UCS 100, GSI 85, B 22
        fuera_rango = (ucs > 100 or gsi > 85 or gsi < 10 or b > 22 or b < 4.5)
        if fuera_rango:
            st.warning("⚠️ **AVISO DE EXTRAPOLACIÓN:** Los valores introducidos exceden el rango de entrenamiento. El modelo estima la tendencia, pero la incertidumbre es mayor.")

    with res2:
        # Indicador visual de confianza basado en la desviación estándar
        conf = max(0, 100 - (sigma * 100))
        st.metric("Índice de Fiabilidad", f"{conf:.1f}%")
        st.progress(conf/100)

    # Registro en Historial
    st.session_state.hist.insert(0, {
        "UCS": ucs, "GSI": gsi, "mo": mo, "B": b, "Peso": v_pp, 
        "Dilat.": v_dil, "Forma": v_for, "Rugos.": v_rug, "Ph (MPa)": round(ph, 3)
    })

# Mostrar Historial si existe
if st.session_state.hist:
    st.divider()
    st.subheader("📜 Historial de Cálculos")
    st.dataframe(pd.DataFrame(st.session_state.hist), use_container_width=True, hide_index=True)
    if st.button("Limpiar Historial"): 
        st.session_state.hist = []
        st.rerun()

st.divider()
st.caption("Arquitectura: Ensemble de Árboles Bayesianos (BART-Inspired) | Capacidad de Extrapolación Moderada Habilitada | Desarrollo para Tesis Doctoral")