import streamlit as st
import pandas as pd
import numpy as np
import arviz as az

# 1. INICIALIZACIÓN DEL ESTADO
if "historial" not in st.session_state:
    st.session_state["historial"] = []

st.set_page_config(page_title="Simulador Ph BART Profesional", layout="wide")

# 2. CARGA DEL MOTOR (Solo lectura de Inferencia)
@st.cache_resource
def load_inference_data():
    try:
        return az.from_netcdf("motor_bart_inferencia.nc")
    except Exception as e:
        st.error(f"❌ Error al cargar 'motor_bart_inferencia.nc': {e}")
        st.stop()

idata = load_inference_data()

st.title("🚀 Predictor Ph - Metamodelo de Alta Fidelidad")
st.markdown("Cálculo basado en la distribución posterior de árboles bayesianos.")

# 3. INTERFAZ DE USUARIO
with st.form("main_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧪 Variables Analíticas")
        mo = st.number_input("Parámetro mo", 5.0, 32.0, 20.0)
        ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0)
        gsi = st.number_input("GSI", 10.0, 85.0, 50.0)
    with col2:
        st.subheader("⚙️ Variables No Analíticas")
        b = st.number_input("Ancho B (m)", 4.5, 22.0, 11.0)
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"])
        v_dil = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)
        v_for = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=1)
        v_rug = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=0)

    submit = st.form_submit_button("🎯 CALCULAR PREDICCIÓN", use_container_width=True)

# 4. LÓGICA DE CÁLCULO ESTABLE
if submit:
    with st.spinner("Calculando respuesta de la superficie..."):
        # Extraemos la variable 'mu' del posterior
        # mu representa el valor esperado del log(Ph + 1)
        mu_samples = idata.posterior["mu"].values.flatten()
        
        # Filtramos valores nulos por seguridad
        mu_samples = mu_samples[~np.isnan(mu_samples)]
        
        # Calculamos el valor predicho (Media de la distribución)
        # IMPORTANTE: Aplicamos la transformación inversa del log1p
        ph_log_mean = np.mean(mu_samples)
        ph_final = np.expm1(ph_log_mean)
        
        # Calculamos la incertidumbre real del modelo (Desviación estándar de Ph)
        ph_samples_real = np.expm1(mu_samples)
        hdi_low = np.percentile(ph_samples_real, 2.5)
        hdi_high = np.percentile(ph_samples_real, 97.5)
        incertidumbre = (hdi_high - hdi_low) / 2

    # --- RESULTADOS ---
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if ph_final > 0:
            st.success(f"### Ph Predicho: **{ph_final:.4f} MPa**")
            st.write(f"**Rango de confianza (95%):** [{hdi_low:.2f} - {hdi_high:.2f}] MPa")
        else:
            st.warning("⚠️ El modelo requiere una recalibración de escala. Revisa las unidades del Excel.")
    
    with res_col2:
        st.metric("Incertidumbre (±)", f"{incertidumbre:.4f} MPa")

    # Registro en historial
    nuevo_registro = {
        "mo": mo, "B": b, "UCS": ucs, "GSI": gsi,
        "Peso": v_pp, "Dilat.": v_dil, "Forma": v_for, "Rugos.": v_rug,
        "Ph (MPa)": round(ph_final, 4), "Err": round(incertidumbre, 4)
    }
    st.session_state["historial"].insert(0, nuevo_registro)

# 5. HISTORIAL
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial de Simulaciones")
    st.table(pd.DataFrame(st.session_state["historial"]))
