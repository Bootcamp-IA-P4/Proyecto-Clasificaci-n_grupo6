import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import COLORS

def set_page_style():
    """
    Configura el estilo general de la página con CSS personalizado.
    """
    st.markdown(f"""
    <style>
        /* Importar fuentes Google */
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500&display=swap');
        
        /* Colores y estilos generales */
        .main {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
            font-family: 'Roboto', sans-serif;
        }}
        .st-bx {{
            background-color: {COLORS['background']};
        }}
        
        /* Estilos para títulos */
        h1, h2, h3, h4, h5 {{
            font-family: 'Google Sans', 'Roboto', sans-serif;
            font-weight: 400;
            color: {COLORS['text']};
        }}
        
        h1 {{
            font-size: 24px;
            margin-bottom: 16px;
        }}
        
        h2 {{
            font-size: 20px;
            margin-bottom: 12px;
        }}
        
        h3 {{
            font-size: 16px;
            margin-bottom: 8px;
        }}
        
        /* Estilos para tarjetas de métricas */
        .metric-card {{
            background-color: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 2px rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 500;
            color: {COLORS['primary']};
            font-family: 'Google Sans', 'Roboto', sans-serif;
        }}
        .metric-label {{
            font-size: 14px;
            color: {COLORS['secondary']};
            font-weight: 400;
        }}
        
        /* Estilos para tarjetas de recomendación */
        .recommendation-card {{
            background-color: white;
            border-left: 4px solid {COLORS['primary']};
            border-radius: 4px;
            padding: 16px;
            margin: 8px 0;
            box-shadow: 0 1px 2px rgba(60,64,67,0.3);
        }}
        .recommendation-title {{
            font-weight: 500;
            font-family: 'Google Sans', 'Roboto', sans-serif;
            color: {COLORS['primary']};
            margin-bottom: 8px;
            font-size: 16px;
        }}
        
        /* Estilo para tablas */
        .dataframe {{
            font-size: 14px;
            border-collapse: collapse;
            width: 100%;
        }}
        .dataframe th {{
            background-color: {COLORS['light_gray']};
            color: {COLORS['secondary']};
            font-weight: 500;
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid {COLORS['border']};
        }}
        .dataframe td {{
            border-bottom: 1px solid {COLORS['border']};
            padding: 8px;
        }}
        
        /* Personalizar selectbox */
        div[data-baseweb="select"] {{
            margin-top: 0.5rem;
        }}
        
        /* Ocultar elementos de navegación nativos de Streamlit */
        [data-testid="stSidebarNav"] {{
            display: none !important;
        }}
        
        /* Estilo para los botones de navegación personalizados */
        .stButton > button[data-testid="StyledFullScreenButton"] {{
            font-family: 'Google Sans', 'Roboto', sans-serif;
            font-weight: 500;
        }}
        
        /* Estilos para botones de sidebar */
        [data-testid="baseButton-secondary"] {{
            background-color: {COLORS['sidebar_inactive']};
            color: {COLORS['text']};
            border: none;
            font-family: 'Google Sans', 'Roboto', sans-serif;
            font-weight: 400;
        }}
        
        [data-testid="baseButton-secondary"]:hover {{
            background-color: {COLORS['sidebar_hover']};
        }}
        
        [data-testid="baseButton-primary"] {{
            background-color: {COLORS['sidebar_active']};
            color: white;
            border: none;
            font-family: 'Google Sans', 'Roboto', sans-serif;
            font-weight: 400;
        }}
        
        /* Estilo para sliders y controles */
        .stSlider {{
            padding-top: 1rem;
            padding-bottom: 1.5rem;
        }}
        
        /* Ajustar tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 40px;
            border-radius: 4px 4px 0 0;
            padding: 0 16px;
            font-size: 14px;
        }}
        
        /* Estilo para expansores */
        .stExpander {{
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
        }}
        
        /* Estilos para sidebar */
        .css-6qob1r {{
            background-color: {COLORS['background']};
        }}
        
        /* Botones principales */
        .stButton > button {{
            font-family: 'Google Sans', 'Roboto', sans-serif;
            font-weight: 500;
            background-color: {COLORS['primary']};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
        }}
        
        .stButton > button:hover {{
            background-color: #1967d2;
        }}
        
        /* Mejorar apariencia de selectores */
        .stSelectbox label, .stSlider label {{
            font-size: 14px;
            color: {COLORS['secondary']};
            font-weight: 500;
        }}
        
        /* Mejorar apariencia de widgets */
        .stCheckbox label p {{
            font-size: 14px;
        }}
    </style>
    """, unsafe_allow_html=True)

def custom_sidebar_navigation():
    """
    Crea botones de navegación personalizados en la barra lateral con selección visual inmediata.
    """
    # Inicializar el estado de sesión para la página seleccionada
    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard"
    
    # Crear los botones directamente
    dashboard_button = st.sidebar.button(
        "Dashboard",
        key="btn_Dashboard",
        use_container_width=True,
        type="primary" if st.session_state.page == "Dashboard" else "secondary"
    )
    
    simulator_button = st.sidebar.button(
        "Simulador",
        key="btn_Simulador",
        use_container_width=True,
        type="primary" if st.session_state.page == "Simulador" else "secondary"
    )
    
    history_button = st.sidebar.button(
        "Historial",
        key="btn_Historial",
        use_container_width=True,
        type="primary" if st.session_state.page == "Historial" else "secondary"
    )
    
    # Manejar los clics con actualización visual inmediata
    if dashboard_button and st.session_state.page != "Dashboard":
        st.session_state.page = "Dashboard"
        st.rerun()
    elif simulator_button and st.session_state.page != "Simulador":
        st.session_state.page = "Simulador"
        st.rerun()
    elif history_button and st.session_state.page != "Historial":
        st.session_state.page = "Historial"
        st.rerun()
    
    # Retornar la página seleccionada
    return st.session_state.page

def metric_card(title, value, description=None, delta=None, delta_color="normal"):
    """
    Muestra una tarjeta de métrica estilizada al estilo de Google Analytics.
    
    Args:
        title (str): Título de la métrica
        value (str): Valor principal
        description (str, optional): Descripción adicional
        delta (str, optional): Valor de cambio
        delta_color (str): Color del delta (normal, good, bad)
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(
            label=title,
            value=value,
            delta=delta,
            delta_color=delta_color
        )
    if description:
        with col2:
            st.caption(description)

def styled_container(content_function, border_color=None):
    """
    Crea un contenedor estilizado con borde de color opcional.
    
    Args:
        content_function: Función que define el contenido
        border_color (str, optional): Color del borde
    """
    if border_color:
        st.markdown(f"""
        <div style="border-left: 5px solid {border_color}; padding-left: 10px;">
        """, unsafe_allow_html=True)
    
    content_function()
    
    if border_color:
        st.markdown("</div>", unsafe_allow_html=True)

def create_bar_chart(data, x, y, title=None, color=None, ax=None, **kwargs):
    """
    Crea un gráfico de barras estilizado.
    
    Args:
        data (DataFrame): Datos para el gráfico
        x (str): Columna para el eje X
        y (str): Columna para el eje Y
        title (str, optional): Título del gráfico
        color (str, optional): Color de las barras
        ax: Eje de matplotlib (opcional)
        **kwargs: Argumentos adicionales para seaborn
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if color is None:
        color = COLORS['chart_blue']
    
    sns.barplot(x=x, y=y, data=data, ax=ax, color=color, **kwargs)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='normal', fontfamily='sans-serif')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['border'])
    ax.spines['bottom'].set_color(COLORS['border'])
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=COLORS['border'])
    
    # Estilo de fuente similar a Google Analytics
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontfamily('sans-serif')
        text.set_fontsize(11)
    
    ax.xaxis.label.set_fontfamily('sans-serif')
    ax.yaxis.label.set_fontfamily('sans-serif')
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    
    return ax

def create_line_chart(data, title=None, ax=None, colors=None, **kwargs):
    """
    Crea un gráfico de líneas estilizado.
    
    Args:
        data (DataFrame): Datos para el gráfico
        title (str, optional): Título del gráfico
        ax: Eje de matplotlib (opcional)
        colors (list, optional): Lista de colores
        **kwargs: Argumentos adicionales para plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    if colors is None:
        colors = [COLORS['chart_blue'], COLORS['orange'], COLORS['teal'], COLORS['purple']]
    
    data.plot(ax=ax, color=colors[:len(data.columns)], **kwargs)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='normal', fontfamily='sans-serif')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['border'])
    ax.spines['bottom'].set_color(COLORS['border'])
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=COLORS['border'])
    
    # Estilo de fuente similar a Google Analytics
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontfamily('sans-serif')
        text.set_fontsize(11)
    
    ax.xaxis.label.set_fontfamily('sans-serif')
    ax.yaxis.label.set_fontfamily('sans-serif')
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    
    return ax

def display_recommendations(recommendations):
    """
    Muestra tarjetas de recomendaciones estilizadas.
    
    Args:
        recommendations (list): Lista de diccionarios con recomendaciones
    """
    if not recommendations:
        return
    
    for rec in recommendations:
        title = rec.get("title", "")
        content = rec.get("content", [])
        
        st.markdown(f"""
        <div class="recommendation-card">
            <div class="recommendation-title">{title}</div>
        """, unsafe_allow_html=True)
        
        for item in content:
            st.markdown(f"<div style='margin-bottom: 8px; font-size: 14px;'>{item}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def sidebar_header(title, description=None):
    """
    Muestra un encabezado estilizado para la barra lateral.
    
    Args:
        title (str): Título del encabezado
        description (str, optional): Descripción adicional
    """
    st.sidebar.markdown(f"<h3 style='color: {COLORS['primary']}; font-family: \"Google Sans\", sans-serif; font-weight: 400; font-size: 18px;'>{title}</h3>", unsafe_allow_html=True)
    if description:
        st.sidebar.markdown(f"<p style='color: {COLORS['secondary']}; font-size: 14px;'>{description}</p>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

def create_download_button(dataframe, filename, label="Descargar datos"):
    """
    Crea un botón de descarga estilizado para un DataFrame.
    
    Args:
        dataframe (DataFrame): DataFrame a descargar
        filename (str): Nombre del archivo a descargar
        label (str): Etiqueta del botón
    """
    csv = dataframe.to_csv(index=False)
    st.download_button(
        label=f"Descargar {label}",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )