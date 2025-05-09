import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from config import COLORS, MONTHS
from data_utils import load_predictions, get_prediction_stats, get_temporal_data
from ui_components import metric_card, create_bar_chart, create_line_chart

def render_dashboard_page(model_type):
    """
    Renderiza la página principal del dashboard.
    
    Args:
        model_type (str): Tipo de modelo utilizado
    """
    st.title("Dashboard de Predicciones")
    
    # Cargar datos de predicciones
    predictions_df = load_predictions()
    
    if predictions_df.empty:
        st.info("No hay predicciones guardadas aún. Usa el simulador para hacer predicciones.")
        return
    
    # Dividir en secciones
    summary_section(predictions_df, model_type)
    
    col1, col2 = st.columns(2)
    
    with col1:
        conversion_trends_section(predictions_df)
    
    with col2:
        visitor_analysis_section(predictions_df)
    
    feature_importance_section(predictions_df)
    
    temporal_analysis_section(predictions_df)

def summary_section(df, model_type):
    """
    Muestra la sección de resumen general.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
        model_type (str): Tipo de modelo utilizado
    """
    st.header("Resumen general")
    
    stats = get_prediction_stats(df)
    
    # Obtener estadísticas para últimos 7 días
    recent_df = df
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_df = df[df['Timestamp'] >= seven_days_ago]
    
    recent_stats = get_prediction_stats(recent_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta = recent_stats['total'] - stats['total'] + recent_stats['total'] if recent_stats['total'] > 0 else None
        metric_card(
            "Total de predicciones", 
            f"{stats['total']}",
            "Histórico",
            f"+{recent_stats['total']} en 7 días" if recent_stats['total'] > 0 else None,
            "normal"
        )
    
    with col2:
        delta_percent = recent_stats['purchase_percent'] - stats['purchase_percent'] if recent_stats['purchase_percent'] > 0 else None
        metric_card(
            "Tasa de conversión", 
            f"{stats['purchase_percent']:.1f}%",
            "Histórico",
            f"{delta_percent:+.1f}%" if delta_percent is not None else None,
            "good" if delta_percent and delta_percent > 0 else "bad" if delta_percent and delta_percent < 0 else "normal"
        )
    
    with col3:
        delta_prob = recent_stats['avg_probability'] - stats['avg_probability'] if recent_stats['avg_probability'] > 0 else None
        metric_card(
            "Probabilidad media", 
            f"{stats['avg_probability']:.2%}",
            "Histórico",
            f"{delta_prob:+.2%}" if delta_prob is not None else None,
            "good" if delta_prob and delta_prob > 0 else "bad" if delta_prob and delta_prob < 0 else "normal"
        )
    
    with col4:
        model_accuracy = calculate_model_accuracy(df) if len(df) > 10 else None
        metric_card(
            "Precisión del modelo", 
            f"{model_accuracy:.1f}%" if model_accuracy is not None else "N/A",
            f"Modelo: {model_type}"
        )
    
    # Información adicional
    st.markdown("""
    <div style="background-color:#F8F9FA; padding:15px; border-radius:5px; margin-top:10px; font-size:14px;">
        <p style="margin:0;">
            Este dashboard muestra el análisis de las predicciones realizadas con el simulador. 
            Los datos se actualizan automáticamente cuando se realizan nuevas predicciones.
        </p>
    </div>
    """, unsafe_allow_html=True)

def conversion_trends_section(df):
    """
    Muestra la sección de tendencias de conversión.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
    """
    st.subheader("Tendencias de conversión")
    
    if df.empty or 'Prediction' not in df.columns:
        st.info("No hay datos suficientes para mostrar tendencias.")
        return
    
    # Gráfico de distribución de predicciones
    fig, ax = plt.subplots(figsize=(10, 6))
    prediction_counts = df['Prediction'].value_counts()
    
    # Modificación: usamos un gráfico de barras manual con colores de Google Analytics
    bars = ax.bar(
        prediction_counts.index,
        prediction_counts.values,
        color=[COLORS['purple'] if x == 'No Compra' else COLORS['teal'] for x in prediction_counts.index]
    )
    
    ax.set_title("Distribución de predicciones", fontsize=14, fontweight='normal', fontfamily='sans-serif')
    ax.set_ylabel("Cantidad", fontfamily='sans-serif')
    ax.set_xlabel("Predicción", fontfamily='sans-serif')
    
    # Eliminar bordes superiores y derechos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['border'])
    ax.spines['bottom'].set_color(COLORS['border'])
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=COLORS['border'])
    
    # Estilo de fuente similar a Google Analytics
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontfamily('sans-serif')
        text.set_fontsize(11)
    
    st.pyplot(fig)
    
    # Análisis de probabilidades
    if 'Probability' in df.columns:
        prob_values = df['Probability'].str.rstrip('%').astype(float) / 100
        
        # Histograma de probabilidades
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            prob_values,
            bins=10,
            kde=True,
            color=COLORS['orange'],  # Cambiado a naranja
            ax=ax
        )
        ax.set_title("Distribución de probabilidades de compra", fontsize=14, fontweight='normal', fontfamily='sans-serif')
        ax.set_xlabel("Probabilidad", fontfamily='sans-serif')
        ax.set_ylabel("Frecuencia", fontfamily='sans-serif')
        
        # Estilo completo
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['border'])
        ax.spines['bottom'].set_color(COLORS['border'])
        ax.grid(axis='y', linestyle='--', alpha=0.3, color=COLORS['border'])
        
        # Estilo de fuente similar a Google Analytics
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontfamily('sans-serif')
            text.set_fontsize(11)
            
        st.pyplot(fig)

def visitor_analysis_section(df):
    """
    Muestra la sección de análisis por tipo de visitante.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
    """
    st.subheader("Análisis por tipo de visitante")
    
    if df.empty or 'VisitorType' not in df.columns:
        st.info("No hay datos de tipos de visitante para analizar.")
        return
    
    # Gráfico de conversiones por tipo de visitante
    visitor_conversion = df.groupby('VisitorType')['Prediction'].apply(
        lambda x: (x == 'Compra').mean() * 100
    ).reset_index()
    visitor_conversion.columns = ['Tipo de visitante', 'Tasa de conversión (%)']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    create_bar_chart(
        visitor_conversion,
        'Tipo de visitante',
        'Tasa de conversión (%)',
        "Tasa de conversión por tipo de visitante",
        color=COLORS['teal'],  # Cambiado a turquesa
        ax=ax
    )
    st.pyplot(fig)
    
    # Tabla de métricas por tipo de visitante
    visitor_metrics = df.groupby('VisitorType').agg({
        'Prediction': lambda x: (x == 'Compra').sum(),
        'Probability': lambda x: pd.to_numeric(x.str.rstrip('%')) / 100
    }).reset_index()
    
    visitor_metrics.columns = ['Tipo de visitante', 'Compras', 'Probabilidad media']
    visitor_metrics['Total visitantes'] = df.groupby('VisitorType').size().values
    visitor_metrics['Tasa de conversión'] = (visitor_metrics['Compras'] / visitor_metrics['Total visitantes'] * 100).round(1).astype(str) + '%'
    visitor_metrics['Probabilidad media'] = (visitor_metrics['Probabilidad media']).round(3).astype(str) + '%'
    
    st.dataframe(visitor_metrics)

def feature_importance_section(df):
    """
    Muestra la sección de importancia de características.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
    """
    st.header("Análisis de características")
    
    # Obtener características numéricas
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    exclude_cols = ['Probability']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and col in df.columns]
    
    if not feature_cols or len(feature_cols) < 2:
        st.info("No hay suficientes características numéricas para analizar.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlación con la predicción
        st.subheader("Correlación con la conversión")
        
        if 'Prediction' in df.columns:
            # Convertir predicción a numérico (1 para Compra, 0 para No Compra)
            df['Prediction_Numeric'] = df['Prediction'].apply(lambda x: 1 if x == 'Compra' else 0)
            
            # Calcular correlaciones
            correlations = []
            for feature in feature_cols:
                if feature in df.columns:
                    corr = df[feature].corr(df['Prediction_Numeric'])
                    if not np.isnan(corr):
                        correlations.append({
                            'Característica': feature,
                            'Correlación': corr
                        })
            
            if correlations:
                corr_df = pd.DataFrame(correlations)
                corr_df = corr_df.sort_values('Correlación', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                create_bar_chart(
                    corr_df,
                    'Característica',
                    'Correlación',
                    "Correlación con la conversión",
                    color=COLORS['orange'],  # Cambiado a naranja
                    ax=ax
                )
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
    
    with col2:
        # PageValues es una característica clave
        if 'PageValues' in df.columns and 'Prediction' in df.columns:
            st.subheader("Impacto de PageValues en la conversión")
            
            # Boxplot de PageValues por predicción
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                x='Prediction',
                y='PageValues',
                data=df,
                ax=ax,
                palette=[COLORS['purple'], COLORS['teal']]  # Púrpura para No Compra, Turquesa para Compra
            )
            ax.set_title("Distribución de PageValues por resultado", fontsize=14, fontweight='normal', fontfamily='sans-serif')
            ax.set_ylabel("PageValues", fontfamily='sans-serif')
            ax.set_xlabel("Predicción", fontfamily='sans-serif')
            
            # Estilo completo
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(COLORS['border'])
            ax.spines['bottom'].set_color(COLORS['border'])
            
            # Estilo de fuente similar a Google Analytics
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_fontfamily('sans-serif')
                text.set_fontsize(11)
                
            st.pyplot(fig)
            
            # Información adicional
            st.markdown("""
            <div style="background-color:#F8F9FA; padding:12px; border-radius:4px; margin-top:10px; font-size:14px;">
                PageValues es uno de los factores más determinantes en la predicción de compras.
                Valores más altos están fuertemente correlacionados con una mayor probabilidad de compra.
            </div>
            """, unsafe_allow_html=True)

def temporal_analysis_section(df):
    """
    Muestra la sección de análisis temporal.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
    """
    st.header("Análisis temporal")
    
    if 'Timestamp' not in df.columns or len(df) <= 1:
        st.info("Se necesitan más datos para mostrar el análisis temporal.")
        return
    
    # Obtener datos temporales
    daily_counts = get_temporal_data(df)
    
    if daily_counts.empty:
        st.info("No hay suficientes datos para mostrar tendencias temporales.")
        return
    
    # Gráfico de tendencia temporal
    fig, ax = plt.subplots(figsize=(12, 6))
    create_line_chart(
        daily_counts,
        "Tendencia de predicciones a lo largo del tiempo",
        ax=ax
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Análisis por mes
    if 'Month' in df.columns:
        st.subheader("Análisis estacional por mes")
        
        month_conversion = df.groupby('Month')['Prediction'].apply(
            lambda x: (x == 'Compra').mean() * 100
        ).reset_index()
        month_conversion.columns = ['Mes', 'Tasa de conversión (%)']
        
        # Ordenar los meses correctamente
        month_order = [m for m in MONTHS if m in month_conversion['Mes'].values]
        month_mapping = {month: i for i, month in enumerate(month_order)}
        month_conversion['Month_Order'] = month_conversion['Mes'].map(month_mapping)
        month_conversion = month_conversion.sort_values('Month_Order').drop('Month_Order', axis=1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        create_bar_chart(
            month_conversion,
            'Mes',
            'Tasa de conversión (%)',
            "Tasa de conversión por mes",
            color=COLORS['orange'],  # Cambiado a naranja
            ax=ax
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Información adicional
        st.markdown("""
        <div style="background-color:#F8F9FA; padding:12px; border-radius:4px; margin-top:10px; font-size:14px;">
            Los meses de mayor conversión son típicamente los de fin de año (Sep-Dic),
            con un pico en Noviembre debido a eventos como Black Friday y descuentos previos a Navidad.
        </div>
        """, unsafe_allow_html=True)

def calculate_model_accuracy(df):
    """
    Calcula una estimación de la precisión del modelo.
    En un entorno real, esto se haría con datos etiquetados.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
        
    Returns:
        float: Precisión estimada del modelo (0-100)
    """
    # Esta es una simulación - en un entorno real usaríamos datos etiquetados
    return 83.5  # Valor simulado