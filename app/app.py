import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import PAGE_CONFIG, APP_INFO
from model_utils import load_model_and_preprocessor
from ui_components import set_page_style, custom_sidebar_navigation
from pages.dashboard import render_dashboard_page
from pages.simulator import render_simulator_page
from pages.history import render_history_page

# Configurar página
st.set_page_config(**PAGE_CONFIG)

# Aplicar estilos CSS personalizados
set_page_style()

# Función principal
def main():
    """
    Función principal que orquesta la aplicación.
    """
    # Cargar recursos (modelo, preprocesador, etc.)
    with st.spinner("Cargando modelo y recursos..."):
        model, preprocessor, expected_features, feature_importance = load_model_and_preprocessor()
    
    # Verificar que los recursos se cargaron correctamente
    if model is None:
        st.error("No se pudo cargar el modelo. Verifica las rutas y el formato del archivo.")
        st.stop()
    
    # Sidebar para navegación
    with st.sidebar:
        # Logo centrado y más grande encima del título
        image_path = os.path.join(os.path.dirname(__file__), "assets", "images", "datashop_analytics.png")
        if os.path.exists(image_path):
            # Crear contenedor para centrar la imagen
            st.markdown("""
            <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
            """, unsafe_allow_html=True)
            st.image(image_path, width=300)  
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Emoji centrado como fallback
            st.markdown("""
            <div style="display: flex; justify-content: center; margin-bottom: 1rem; font-size: 40px;">
            🛒
            </div>
            """, unsafe_allow_html=True)
        
        
        # Usar navegación personalizada con botones personalizados
        selected_page = custom_sidebar_navigation()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Sobre esta aplicación")
        
        # Mostrar el tipo de modelo correcto
        model_type = APP_INFO.get("model_type", "Machine Learning")
        
        # Descripción del negocio
        st.sidebar.markdown(f"""
        ### Optimización de Conversiones en E-Commerce
        
        Esta herramienta utiliza un modelo de **{model_type}** para predecir la probabilidad 
        de que un usuario realice una compra basándose en su comportamiento de navegación.
        
        **Aplicaciones de negocio:**
        
        - **Personalización en tiempo real** de la experiencia del usuario
        - **Optimización de campañas** en periodos de alta conversión
        - **Mejora del diseño del sitio** para reducir abandonos
        - **Estrategias de retargeting** más efectivas
        - **Asignación eficiente** de recursos de marketing
        
        La tasa actual de conversión del sitio es del **{APP_INFO.get('conversion_rate', '15.6%')}**, con grandes 
        variaciones estacionales y diferencias entre tipos de visitantes.
        """)
    
    # Renderizar la página correspondiente
    if selected_page == "Dashboard":
        render_dashboard_page(model_type)
    elif selected_page == "Simulador":
        render_simulator_page(model, preprocessor, expected_features, feature_importance)
    elif selected_page == "Historial":
        render_history_page()

# Ejecutar la aplicación
if __name__ == "__main__":
    main()