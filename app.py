import streamlit as st
import pandas as pd
import numpy as np
import arviz as az
import pymc_bart as pmb

# ==============================================================================
# 1. INICIALIZACIÓN DEL ESTADO (DEBE IR ANTES QUE NADA)
# ==============================================================================
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# ==============================================================================
# 2. CONFIGURACIÓN Y CARGA DEL MOTOR (.NC)
# ==============================================================================
st.set_page_config(page_title="Simulador Ph BART - Doctorado", layout="wide")

@st.cache_resource
def load_engine():
    # Asegúrate de que este nombre sea EXACTO al de GitHub
    return az.from_netcdf("modelo_bart_final.nc")

try:
    idata = load_engine()
except Exception as e:
    st.error(f"❌ Error al cargar el motor bayesiano: {e}")
    st.info("Comprueba que 'modelo_bart_final.nc' esté en la raíz de tu repositorio de GitHub.")
    st.stop()

# ==============================================================================
# 3. INTERFAZ DE USUARIO (TU MÁSCARA)
# ==============================================================================
st.title("🚀 Predictor Ph - Motor Bayesiano BART")
st.markdown("Inferencia mediante árboles aditivos bayesianos para superficies de respuesta continuas.")

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

# ==============================================================================
# 4. LÓGICA DE CÁLCULO
# ==============================================================================
if submit:
    # Mapeo numérico
    pp_val = 1.0 if v_pp == "Con Peso" else 0.0
    dil_val = 1.0 if v_dil == "Asociada" else 0.0
    for_val = 1.0 if v_for == "Axisimétrica" else 0.0
    rug_val = 1.0 if v_rug == "Rugoso" else 0.0
    
    # Vector: mo, B, UCS, GSI, PP, Dil, Form, Rug (Orden según tu imagen)
    vec = [mo, b, ucs, gsi, pp_val, dil_val, for_val, rug_val]
    
    with st.spinner("Realizando inferencia bayesiana..."):
        # Extraemos muestras de la posterior y transformamos a escala real
        # Hacemos el expm1 sobre todas las muestras ANTES de calcular estadísticas
        mu_real_scale = np.expm1(idata.posterior["mu"].values)
        
        # Predicción central (Mediana, más estable ante colas largas)
        ph_resultado = np.median(mu_real_scale)
        
        # Incertidumbre real (HDI 95%)
        hdi_low = np.percentile(mu_real_scale, 2.5)
        hdi_high = np.percentile(mu_real_scale, 97.5)
        error_std = (hdi_high - hdi_low) / 2

    # Presentación
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.success(f"### Ph Predicho: **{ph_resultado:.4f} MPa**")
        st.write(f"**Intervalo de Credibilidad (95%):** [{hdi_low:.2f} - {hdi_high:.2f}] MPa")
    
    with res_col2:
        st.metric("Incertidumbre", f"± {error_std:.4f} MPa")

    # Guardar en historial
    nuevo_registro = {
        "UCS": ucs, "GSI": gsi, "mo": mo, "B": b,
        "Peso": v_pp, "Dilat.": v_dil, "Forma": v_for, "Rugos.": v_rug,
        "Ph (MPa)": round(ph_resultado, 4),
        "Incertidumbre (±)": round(error_std, 4)
    }
    st.session_state["historial"].insert(0, nuevo_registro)

# ==============================================================================
# 5. MOSTRAR HISTORIAL (SEGURO ANTE REINICIOS)
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
st.caption("BART Engine | Doctorado | Inferencia Bayesiana sobre NetCDF")
