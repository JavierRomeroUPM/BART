import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
import pymc_bart as pmb
import arviz as az

# 1. INICIALIZACIÓN DEL HISTORIAL
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# 2. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Simulador Ph BART Profesional", layout="wide")

# 3. CARGA Y RECONSTRUCCIÓN DEL MOTOR
@st.cache_resource
def reconstruir_motor():
    # Cargamos la inferencia del archivo .nc
    idata = az.from_netcdf("motor_bart_inferencia.nc")
    
    # Datos dummy para inicializar la estructura del modelo (8 columnas)
    X_dummy = np.zeros((1, 8))
    y_dummy = np.zeros(1)
    
    with pm.Model() as model:
        # Contenedor de datos actualizable
        X_obs = pm.Data("X_obs", X_dummy)
        
        # Estructura BART idéntica a la de Colab
        mu = pmb.BART("mu", X_obs, y_dummy, m=50)
        sigma = pm.HalfNormal("sigma", sigma=1)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_dummy)
        
    return model, idata

model, idata = reconstruir_motor()

# 4. INTERFAZ DE USUARIO (MÁSCARA PROFESIONAL)
st.title("🚀 Predictor Ph Dinámico - Metamodelo BART")
st.markdown("Sistema de alta fidelidad con reconstrucción de superficie de respuesta en tiempo real.")

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

    submit = st.form_submit_button("🎯 CALCULAR PREDICCIÓN ACTUALIZADA", use_container_width=True)

# 5. LÓGICA DE CÁLCULO DINÁMICO
if submit:
    # Mapeo numérico para el modelo
    pp_val = 1.0 if v_pp == "Con Peso" else 0.0
    dil_val = 1.0 if v_dil == "Asociada" else 0.0
    for_val = 1.0 if v_for == "Axisimétrica" else 0.0
    rug_val = 1.0 if v_rug == "Rugoso" else 0.0
    
    # Vector de entrada (Orden exacto de tu imagen de Excel)
    # mo, B, UCS, GSI, PP, Dil, Form, Rug
    vec = np.array([[mo, b, ucs, gsi, pp_val, dil_val, for_val, rug_val]])
    
    with st.spinner("Inyectando datos en el metamodelo..."):
        with model:
            # Actualizamos los datos del contenedor
            pm.set_data({"X_obs": vec})
            # Realizamos la predicción usando la memoria del .nc
            ppc = pm.sample_posterior_predictive(idata, var_names=["mu"], progressbar=False)
            
        # Extraemos las muestras de la predicción y transformamos de log1p a MPa
        mu_pred_samples = ppc.posterior_predictive["mu"].values.flatten()
        
        # Resultado final (Media para asegurar suavidad)
        ph_final = np.expm1(np.mean(mu_pred_samples))
        
        # Incertidumbre científica (Intervalo de confianza 95%)
        hdi_low = np.expm1(np.percentile(mu_pred_samples, 2.5))
        hdi_high = np.expm1(np.percentile(mu_pred_samples, 97.5))
        error_std = (hdi_high - hdi_low) / 2

    # --- PRESENTACIÓN DE RESULTADOS ---
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.success(f"### Ph Predicho: **{ph_final:.4f} MPa**")
        st.write(f"**Intervalo de Credibilidad (95%):** [{hdi_low:.2f} - {hdi_high:.2f}] MPa")
    
    with res_col2:
        st.metric("Incertidumbre (±)", f"{error_std:.4f} MPa")
        st.info("💡 **Superficie de Seda**: Inferencia bayesiana completa.")

    # Guardar registro de 10 columnas en el historial
    nuevo_registro = {
        "mo": mo, "B": b, "UCS": ucs, "GSI": gsi,
        "Peso": v_pp, "Dilat.": v_dil, "Forma": v_for, "Rugos.": v_rug,
        "Ph (MPa)": round(ph_final, 4), "Err (±)": round(error_std, 4)
    }
    st.session_state["historial"].insert(0, nuevo_registro)

# 6. HISTORIAL DE SIMULACIONES
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial de Resultados")
    st.dataframe(pd.DataFrame(st.session_state["historial"]), use_container_width=True, hide_index=True)
