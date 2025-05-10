import streamlit as st
import pandas as pd
from datetime import datetime
from config import COLORS, MONTHS
from data_utils import load_predictions, filter_predictions, get_prediction_stats
from ui_components import metric_card, create_download_button

def render_history_page():
    """
    Renderiza la página de historial de predicciones.
    """
    st.title("Historial de Predicciones")
    
    # Cargar el historial de predicciones
    predictions_df = load_predictions()
    
    if predictions_df.empty:
        st.info("No hay predicciones guardadas aún. Usa el simulador para hacer predicciones.")
        return
    
    # Mostrar métricas de resumen
    display_summary_metrics(predictions_df)
    
    # Filtros para el historial
    filtered_df = create_and_apply_filters(predictions_df)
    
    # Mostrar el historial filtrado
    st.subheader("Resultados filtrados")
    
    # Aplicar estilo a la tabla
    st.markdown("""
    <style>
    .dataframe-container {
        background-color: white;
        border-radius: 4px;
        padding: 12px;
        box-shadow: 0 1px 2px rgba(60,64,67,0.3);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mostrar número de resultados
    st.caption(f"Mostrando {len(filtered_df)} de {len(predictions_df)} predicciones totales")
    
    # Opción para descargar el historial
    create_download_button(
        filtered_df,
        "predicciones_filtradas.csv",
        "historial filtrado"
    )

def display_summary_metrics(df):
    """
    Muestra métricas de resumen para el historial de predicciones.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
    """
    stats = get_prediction_stats(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metric_card(
            "Total de predicciones", 
            f"{stats['total']}",
            "Número total de predicciones realizadas"
        )
    
    with col2:
        metric_card(
            "Predicciones de compra", 
            f"{stats['purchases']} ({stats['purchase_percent']:.1f}%)",
            "Predicciones que resultaron en compra"
        )
    
    with col3:
        metric_card(
            "Probabilidad media", 
            f"{stats['avg_probability']:.2%}",
            "Promedio de todas las probabilidades"
        )

def create_and_apply_filters(df):
    """
    Crea y aplica filtros al DataFrame de predicciones.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
        
    Returns:
        DataFrame: DataFrame filtrado
    """
    # Contenedor para los filtros con estilo 
    # Incluimos el título directamente en el contenedor para evitar la barra
    st.markdown("""
    <style>
    .filter-container {
        background-color: #f8f9fa;
        border-radius: 4px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .filter-title {
        font-size: 18px;
        font-weight: 400;
        font-family: 'Google Sans', 'Roboto', sans-serif;
        margin-bottom: 16px;
        color: #202124;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    st.markdown('<div class="filter-title">Filtrar predicciones</div>', unsafe_allow_html=True)
    
    # Hacer una copia del DataFrame original para los filtros
    filtered_df = df.copy()
    
    # Preparar datos de timestamp si están presentes
    if 'timestamp' in filtered_df.columns:
        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
        min_date = filtered_df['timestamp'].min().date()
        max_date = filtered_df['timestamp'].max().date()
        
        st.markdown("<div style='font-weight: 500; margin-top: 10px; margin-bottom: 5px;'>Rango de fechas</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Fecha inicial",
                min_date,
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "Fecha final",
                max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        # Aplicar filtro de fechas
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= start_date) &
            (filtered_df['timestamp'].dt.date <= end_date)
        ]
    
    # Filtros adicionales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'prediction' in filtered_df.columns:
            available_predictions = sorted(filtered_df['prediction'].unique())
            prediction_filter = st.multiselect(
                "Filtrar por resultado",
                options=available_predictions,
                default=available_predictions
            )
            if prediction_filter and len(prediction_filter) < len(available_predictions):
                filtered_df = filtered_df[filtered_df['prediction'].isin(prediction_filter)]
    
    with col2:
        if 'month' in filtered_df.columns:
            available_months = sorted(filtered_df['month'].unique())
            month_filter = st.multiselect(
                "Filtrar por mes",
                options=available_months,
                default=available_months
            )
            if month_filter and len(month_filter) < len(available_months):
                filtered_df = filtered_df[filtered_df['month'].isin(month_filter)]
    
    with col3:
        if 'visitor_type' in filtered_df.columns:
            available_types = sorted(filtered_df['visitor_type'].unique())
            visitor_filter = st.multiselect(
                "Filtrar por tipo de visitante",
                options=available_types,
                default=available_types
            )
            if visitor_filter and len(visitor_filter) < len(available_types):
                filtered_df = filtered_df[filtered_df['visitor_type'].isin(visitor_filter)]
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return filtered_df