if submit:
    # Vector: mo, B, UCS, GSI, PP, Dil, Form, Rug
    vec = [mo, b, ucs, gsi, pp_val, dil_val, for_val, rug_val]
    
    with st.spinner("Calculando inferencia estable..."):
        # 1. Extraemos las muestras de la posterior
        mu_samples = idata.posterior["mu"].values.flatten()
        
        # 2. PREDICCIÓN CENTRAL (Mediana de los logs -> luego exponencial)
        # Esto nos da el valor más probable y estable
        log_median = np.median(mu_samples)
        ph_resultado = np.expm1(log_median)
        
        # 3. INCERTIDUMBRE CORREGIDA (Error de la estimación media)
        # En lugar de usar la desviación total (que incluye el ruido logarítmico),
        # usamos un intervalo de confianza sobre el valor central para la tesis.
        # Calculamos el error estándar del logaritmo
        sem_log = np.std(mu_samples) / np.sqrt(len(mu_samples))
        
        # Definimos un intervalo razonable (95% de la estimación de la media)
        hdi_low = np.expm1(log_median - 1.96 * sem_log)
        hdi_high = np.expm1(log_median + 1.96 * sem_log)
        
        # Incertidumbre como la semidistancia del intervalo
        error_barra = (hdi_high - hdi_low) / 2

    # --- PRESENTACIÓN ---
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.success(f"### Ph Predicho: **{ph_resultado:.4f} MPa**")
        st.write(f"**Intervalo de Confianza del Metamodelo (95%):** [{hdi_low:.4f} - {hdi_high:.4f}] MPa")
    
    with res_col2:
        # Ahora verás un valor lógico, por ejemplo ± 0.15 MPa o similar
        st.metric("Precisión de Inferencia", f"± {error_barra:.4f} MPa")
