import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import COLORS, DEFAULT_USER_DATA, MONTHS, VISITOR_TYPES
from model_utils import preprocess_data, make_prediction
from data_utils import save_prediction
from ui_components import metric_card, display_recommendations
from utils import generate_marketing_recommendations, get_navigation_pattern

def render_simulator_page(model, preprocessor, expected_features, feature_importance):
    """
    Renderiza la página del simulador de comportamiento de usuario.
    
    Args:
        model: Modelo cargado
        preprocessor: Preprocesador cargado
        expected_features: Lista de características esperadas por el modelo
        feature_importance: DataFrame con la importancia de características
    """
    st.title("Simulador de Comportamiento de Usuario")
    
    st.markdown("""
    Este simulador te permite experimentar con diferentes escenarios de comportamiento de usuario 
    para predecir la probabilidad de compra y obtener recomendaciones estratégicas de marketing.
    """)
    
    # Crear el simulador con sliders y controles
    st.subheader("Configura el escenario de usuario")
    
    # Inicializar sim_data con valores por defecto
    sim_data = DEFAULT_USER_DATA.copy()
    
    # Dividir en dos columnas
    col1, col2 = st.columns(2)
    
    with col1:
        # Principales indicadores de conversión
        st.markdown("#### Indicadores principales")
        
        sim_data['PageValues'] = st.slider(
            "PageValues (valor comercial de página)",
            min_value=0.0,
            max_value=50.0,
            value=sim_data['PageValues'],
            step=1.0,  # Cambiado a incrementos de 1.0
            help="El valor asignado a páginas visitadas (mayor valor indica mayor intención de compra)"
        )
        
        sim_data['ExitRates'] = st.slider(
            "ExitRates (tasa de salida)",
            min_value=0.0,
            max_value=0.2,
            value=sim_data['ExitRates'],
            step=0.01,
            help="Porcentaje de salidas desde una página específica"
        )
        
        sim_data['BounceRates'] = st.slider(
            "BounceRates (tasa de rebote)",
            min_value=0.0,
            max_value=0.2,
            value=sim_data['BounceRates'],
            step=0.01,
            help="Porcentaje de visitantes que abandonan el sitio tras ver una sola página"
        )
        
        # Comportamiento de páginas
        st.markdown("#### Páginas visitadas")
        
        sim_data['Administrative'] = st.slider(
            "Páginas administrativas",
            min_value=0,
            max_value=20,
            value=sim_data['Administrative'],
            help="Número de páginas administrativas visitadas (carrito, cuenta, etc.)"
        )
        
        sim_data['Informational'] = st.slider(
            "Páginas informativas",
            min_value=0,
            max_value=20,
            value=sim_data['Informational'],
            help="Número de páginas informativas visitadas (FAQ, políticas, etc.)"
        )
        
        sim_data['ProductRelated'] = st.slider(
            "Páginas de productos",
            min_value=0,
            max_value=100,
            value=sim_data['ProductRelated'],
            help="Número de páginas de productos visitadas"
        )
    
    with col2:
        # Tiempo de navegación
        st.markdown("#### Tiempo de navegación (segundos)")
        
        sim_data['Administrative_Duration'] = st.slider(
            "Tiempo en páginas administrativas",
            min_value=0.0,
            max_value=500.0,
            value=sim_data['Administrative_Duration'],
            step=10.0,
            help="Tiempo dedicado a páginas administrativas en segundos"
        )
        
        sim_data['Informational_Duration'] = st.slider(
            "Tiempo en páginas informativas",
            min_value=0.0,
            max_value=500.0,
            value=sim_data['Informational_Duration'],
            step=10.0,
            help="Tiempo dedicado a páginas informativas en segundos"
        )
        
        sim_data['ProductRelated_Duration'] = st.slider(
            "Tiempo en páginas de productos",
            min_value=0.0,
            max_value=1000.0,
            value=sim_data['ProductRelated_Duration'],
            step=10.0,
            help="Tiempo dedicado a páginas de productos en segundos"
        )
        
        # Contexto de visita
        st.markdown("#### Contexto de la visita")
        
        month_index = MONTHS.index(sim_data['Month']) if sim_data['Month'] in MONTHS else 0
        
        sim_data['Month'] = st.selectbox(
            "Mes de la visita",
            options=MONTHS,
            index=month_index,
            help="Los meses con mayor conversión son: Nov (25.5%), Oct (20.9%), Sep (19.2%)"
        )
        
        visitor_index = VISITOR_TYPES.index(sim_data['VisitorType']) if sim_data['VisitorType'] in VISITOR_TYPES else 0
        
        sim_data['VisitorType'] = st.selectbox(
            "Tipo de visitante",
            options=VISITOR_TYPES,
            index=visitor_index,
            help="Los nuevos visitantes tienen mayor tasa de conversión (24.9%) que los recurrentes (14.1%)"
        )
        
        sim_data['Weekend'] = 1 if st.checkbox(
            "¿Es fin de semana?",
            value=bool(sim_data['Weekend']),
            help="La tasa de conversión es ligeramente mayor en fines de semana (17.5% vs 15.1%)"
        ) else 0
    
    # Calcular métricas derivadas
    total_pages = sim_data['Administrative'] + sim_data['Informational'] + sim_data['ProductRelated']
    total_duration = sim_data['Administrative_Duration'] + sim_data['Informational_Duration'] + sim_data['ProductRelated_Duration']
    
    # Mostrar resumen de navegación
    st.markdown("#### Resumen de navegación")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de páginas visitadas", total_pages)
    with col2:
        st.metric("Tiempo total (segundos)", f"{total_duration:.0f}")
    with col3:
        # Determinar el patrón de navegación
        pattern_name, pattern_desc, pattern_conv = get_navigation_pattern(total_pages, total_duration)
        st.metric("Patrón de navegación", pattern_name, pattern_conv)
    
    # Botón para ejecutar simulación
    if st.button("Ejecutar simulación", use_container_width=True, type="primary"):
        run_simulation(sim_data, model, preprocessor, expected_features, feature_importance)

def run_simulation(sim_data, model, preprocessor, expected_features, feature_importance):
    """
    Ejecuta la simulación y muestra los resultados.
    
    Args:
        sim_data (dict): Datos de entrada del usuario
        model: Modelo cargado
        preprocessor: Preprocesador cargado
        expected_features: Lista de características esperadas por el modelo
        feature_importance: DataFrame con la importancia de características
    """
    with st.spinner("Analizando comportamiento de usuario..."):
        # Preprocesar los datos
        processed_data = preprocess_data(sim_data, preprocessor, expected_features)
        
        # Realizar la predicción
        prediction, probability = make_prediction(model, processed_data)
        
        # Mostrar el resultado
        st.subheader("Resultado de la predicción")
        
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.success("PREDICCIÓN: USUARIO REALIZARÁ UNA COMPRA")
            else:
                st.error("PREDICCIÓN: USUARIO NO REALIZARÁ UNA COMPRA")
        
        with col2:
            st.metric("Probabilidad de compra", f"{probability:.2%}")
        
        # Mostrar barra de progreso con color dependiendo de la probabilidad
        progress_color = get_probability_color(probability)
        st.markdown(
            f"""
            <div style="width:100%; background-color:#f0f2f6; height:20px; border-radius:10px;">
                <div style="width:{probability*100}%; background-color:{progress_color}; height:20px; border-radius:10px;"></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Generar recomendaciones de marketing
        recommendations = generate_marketing_recommendations(sim_data, probability)
        
        # Mostrar recomendaciones
        st.subheader("Recomendaciones estratégicas de marketing")
        
        tabs = st.tabs([rec["title"] for rec in recommendations])
        
        for i, tab in enumerate(tabs):
            with tab:
                for item in recommendations[i]["content"]:
                    st.markdown(item)
        
        # Mostrar características más influyentes
        if feature_importance is not None:
            st.subheader("Factores determinantes en la predicción")
            
            # Obtener las características más importantes y sus valores
            important_indices = processed_data.abs().mean().sort_values(ascending=False).index[:5]
            feature_values = []
            
            for feature in important_indices:
                value = processed_data[feature].iloc[0]
                feature_values.append({
                    'Característica': feature,
                    'Valor': value
                })
            
            # Crear una tabla con estos valores
            feat_df = pd.DataFrame(feature_values)
            st.table(feat_df)
        
        # Guardar predicción automáticamente
        success = save_prediction(sim_data, prediction, probability)
        if success:
            st.success("Predicción guardada automáticamente en el historial")
        else:
            st.warning("No se pudo guardar la predicción. Verifica los permisos de escritura")
        
        # Mostrar datos procesados
        with st.expander("Ver datos procesados"):
            st.subheader("Datos después del preprocesamiento")
            st.dataframe(processed_data)

def get_probability_color(probability):
    """
    Devuelve un color basado en la probabilidad.
    
    Args:
        probability (float): Probabilidad de compra
        
    Returns:
        str: Código de color en formato hexadecimal
    """
    if probability >= 0.7:
        return COLORS['success']   
    elif probability >= 0.4:
        return COLORS['orange']      
    else:
        return COLORS['chart_blue']  