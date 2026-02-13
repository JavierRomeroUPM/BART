import streamlit as st
import pandas as pd
import numpy as np
import arviz as az
import pymc_bart as pmb

# Configuración de página
st.set_page_config(page_title="Calculadora Geotécnica BART", layout="centered")

# 1. CARGA DEL MODELO (Asegúrate de tener el archivo .nc en la carpeta)
@st.cache_resource
def cargar_modelo():
    return az.from_netcdf("modelo_bart_final.nc")

try:
    idata = cargar_modelo()
except:
    st.error("⚠️ No se encuentra el archivo 'modelo_bart_final.nc'. Asegúrate de subirlo a la carpeta de la App.")
    st.stop()

st.title("🏗️ Sistema Experto: Predicción de $P_h$")
st.markdown("### Modelo de Regresión Bayesiana (BART)")

# 2. ENTRADA DE DATOS (VENTANAS NUMÉRICAS Y DESPLEGABLES)
st.subheader("Parámetros de Entrada")

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        ucs = st.number_input("UCS (Resistencia Compresión) [MPa]", value=50.0, format="%.2f")
        gsi = st.number_input("GSI (Geological Strength Index)", value=50.0, format="%.1f")
        mi = st.number_input("Parámetro mi (Hoek-Brown)", value=15.0, format="%.2f")
        d_param = st.number_input("Factor de Daño (D)", value=0.0, min_value=0.0, max_value=1.0, step=0.1)

    with col2:
        gamma = st.number_input("Densidad (γ) [kN/m³]", value=25.0, format="%.2f")
        z = st.number_input("Profundidad (Z) [m]", value=100.0, format="%.1f")
        b_tunel = st.number_input("Ancho de excavación (B) [m]", value=10.0, format="%.2f")
        # Ejemplo de Desplegable Categórico
        sobrecarga = st.selectbox("Nivel de Sobrecarga (S)", 
                                 options=[0, 100, 500, 1000],
                                 help="Seleccione la categoría de presión superficial")

# 3. BOTÓN DE CÁLCULO
if st.button("🚀 CALCULAR PRESIÓN DE HUNDIMIENTO", type="primary", use_container_width=True):
    
    # Preparar el vector (ajusta el orden si en tu Excel era distinto)
    # Orden: [GSI, UCS, mi, D, gamma, Z, B, S]
    X_new = np.array([[gsi, ucs, mi, d_param, gamma, z, b_tunel, sobrecarga]])
    
    with st.spinner("Procesando incertidumbre bayesiana..."):
        # Extraer muestras de la distribución 'mu'
        mu_samples = idata.posterior["mu"]
        
        # Cálculo del valor medio (el punto en la curva suave)
        ph_log_mean = mu_samples.mean().values
        ph_final = np.expm1(ph_log_mean)
        
        # CÁLCULO DE LA INCERTIDUMBRE (Desviación Estándar de las muestras)
        # Esto indica cuánto "dudan" los árboles de BART para esos inputs
        ph_std = mu_samples.std().values
        incertidumbre = (np.expm1(ph_log_mean + ph_std) - np.expm1(ph_log_mean - ph_std)) / 2

    # 4. RECUADRO DE RESULTADOS E INCERTIDUMBRE
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(label="Presión de Hundimiento ($P_h$)", value=f"{ph_final:.3f} MPa")
    
    with res_col2:
        # El recuadro de incertidumbre que pedías
        st.info(f"**Incertidumbre del Modelo:** ± {incertidumbre:.4f} MPa")
        st.caption("Intervalo de confianza basado en la varianza de la posterior (BART).")

    # Guardar en historial (opcional)
    if 'historial' not in st.session_state:
        st.session_state.historial = []
    st.session_state.historial.insert(0, {"Fecha": pd.Timestamp.now(), "Ph": ph_final, "Incertidumbre": incertidumbre})

# 5. HISTORIAL (Simplificado)
if st.checkbox("Ver Historial de Cálculos"):
    st.table(pd.DataFrame(st.session_state.historial))
