import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime
import csv

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Compras Online",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definir rutas de los archivos
model_path = 'notebooks/modeling/xgboost/pkl_exports/xgboost_optimized.pkl'
preprocessor_path = 'notebooks/modeling/xgboost/pkl_exports/xgboost_preprocessor.pkl'
predictions_path = 'data/predictions.csv'

# Función para cargar modelo y preprocesador
@st.cache_resource
def load_resources():
    try:
        # Cargar el modelo
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Obtener características esperadas
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
            else:
                expected_features = []
        else:
            st.sidebar.error(f"No se encontró el modelo en: {model_path}")
            return None, None, [], None
        
        # Cargar el preprocesador
        preprocessor = None
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
        else:
            st.sidebar.warning("No se encontró el preprocesador.")
        
        # Obtener la importancia de características
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': expected_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
        return model, preprocessor, expected_features, feature_importance
    
    except Exception as e:
        st.sidebar.error(f"Error al cargar el modelo o preprocesador: {str(e)}")
        return None, None, [], None

# Cargar recursos
model, preprocessor, expected_features, feature_importance = load_resources()

# Verificar que los recursos se cargaron correctamente
if model is None:
    st.error("No se pudo cargar el modelo. Verifica las rutas y el formato del archivo.")
    st.stop()
else:
    st.sidebar.success(f"Modelo cargado correctamente. Características: {len(expected_features)}")
    if preprocessor:
        st.sidebar.success(f"Preprocesador cargado correctamente.")

# Función para aplicar el preprocesamiento utilizando la información del preprocesador
def preprocess_input_data(input_data):
    # Convertir el diccionario de entrada a un DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Uso de la información del preprocesador si está disponible
    if preprocessor:
        # 1. Transformaciones logarítmicas
        for log_feat in preprocessor['preprocesado_info']['transformaciones_log']:
            log_feat_name = f"{log_feat}_Log"
            input_df[log_feat_name] = np.log1p(input_df[log_feat])
        
        # 2. Variables derivadas
        for deriv_feat, formula in preprocessor['preprocesado_info']['variables_derivadas'].items():
            if deriv_feat == 'TotalPages':
                input_df[deriv_feat] = input_df['Administrative'] + input_df['Informational'] + input_df['ProductRelated']
            elif deriv_feat == 'TotalDuration':
                input_df[deriv_feat] = input_df['Administrative_Duration'] + input_df['Informational_Duration'] + input_df['ProductRelated_Duration']
            elif deriv_feat == 'PageValues_NonZero':
                input_df[deriv_feat] = (input_df['PageValues'] > 0).astype(int)
        
        # 3. Variables dummy para Month
        for month_col in preprocessor['month_columns']:
            month = month_col.replace('Month_', '')
            input_df[month_col] = 1 if input_df['Month'].iloc[0] == month else 0
        
        # 4. Variables dummy para VisitorType
        for visitor_col in preprocessor['visitor_columns']:
            visitor_type = visitor_col.replace('VisitorType_', '')
            input_df[visitor_col] = 1 if input_df['VisitorType'].iloc[0] == visitor_type else 0
    else:
        # Implementación manual de transformaciones si no hay preprocesador
        # 1. Transformaciones logarítmicas
        input_df['Administrative_Duration_Log'] = np.log1p(input_df['Administrative_Duration'])
        input_df['Informational_Duration_Log'] = np.log1p(input_df['Informational_Duration'])
        input_df['ProductRelated_Duration_Log'] = np.log1p(input_df['ProductRelated_Duration'])
        input_df['PageValues_Log'] = np.log1p(input_df['PageValues'])
        
        # 2. Variables derivadas
        input_df['PageValues_NonZero'] = (input_df['PageValues'] > 0).astype(int)
        input_df['TotalPages'] = input_df['Administrative'] + input_df['Informational'] + input_df['ProductRelated']
        input_df['TotalDuration'] = input_df['Administrative_Duration'] + input_df['Informational_Duration'] + input_df['ProductRelated_Duration']
        
        # 3. Variables dummy para Month
        all_months = ['Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in all_months:
            col_name = f'Month_{month}'
            input_df[col_name] = 1 if input_df['Month'].iloc[0] == month else 0
        
        # 4. Variables dummy para VisitorType
        visitor_types = ['New_Visitor', 'Returning_Visitor']
        for vtype in visitor_types:
            col_name = f'VisitorType_{vtype}'
            input_df[col_name] = 1 if input_df['VisitorType'].iloc[0] == vtype else 0
    
    # Seleccionar solo las características que el modelo espera
    final_df = pd.DataFrame()
    for feature in expected_features:
        if feature in input_df.columns:
            final_df[feature] = input_df[feature]
        else:
            final_df[feature] = 0
    
    return final_df

# Función para guardar una predicción en CSV
def save_prediction(input_data, prediction, probability):
    # Crear un diccionario con los datos de la predicción
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_data = {
        'Timestamp': timestamp,
        'Prediction': 'Compra' if prediction == 1 else 'No Compra',
        'Probability': f"{probability:.2%}",
    }
    
    # Añadir los datos de entrada al diccionario
    for key, value in input_data.items():
        prediction_data[key] = value
    
    # Asegurar que existe el directorio
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    
    # Verificar si el archivo existe
    file_exists = os.path.isfile(predictions_path)
    
    # Guardar al CSV
    try:
        # Si el archivo no existe, crear con cabecera
        if not file_exists:
            with open(predictions_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=prediction_data.keys())
                writer.writeheader()
                writer.writerow(prediction_data)
        else:
            # Si existe, verificar columnas y añadir fila
            existing_df = pd.read_csv(predictions_path)
            # Verificar si hay columnas nuevas
            missing_cols = set(prediction_data.keys()) - set(existing_df.columns)
            if missing_cols:
                # Añadir columnas nuevas a los datos existentes
                for col in missing_cols:
                    existing_df[col] = None
            # Añadir nueva fila y guardar
            new_row = pd.DataFrame([prediction_data])
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            updated_df.to_csv(predictions_path, index=False)
        
        return True
    except Exception as e:
        st.error(f"Error al guardar la predicción: {str(e)}")
        return False

# Función para cargar el historial de predicciones
def load_predictions():
    if os.path.exists(predictions_path):
        try:
            return pd.read_csv(predictions_path)
        except Exception as e:
            st.error(f"Error al cargar el historial: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

# Función para generar recomendaciones de marketing basadas en los datos y predicciones
def generate_marketing_recommendations(input_data, probability):
    recommendations = []
    
    # Clasificación de probabilidad
    if probability >= 0.7:
        probability_level = "alta"
    elif probability >= 0.4:
        probability_level = "media"
    else:
        probability_level = "baja"
    
    # Recomendaciones basadas en la probabilidad general
    recommendation = {
        "title": f"Probabilidad {probability_level} de compra ({probability:.2%})",
        "content": []
    }
    
    if probability_level == "alta":
        recommendation["content"] = [
            "✓ Este usuario muestra una alta intención de compra",
            "✓ Maximizar el valor del carrito con recomendaciones complementarias",
            "✓ Ofrecer opciones de envío premium o garantías extendidas",
            "✓ Mostrar productos similares de mayor categoría/precio",
            "✓ Minimizar distracciones en el proceso de checkout"
        ]
    elif probability_level == "media":
        recommendation["content"] = [
            "✓ El usuario muestra interés pero necesita incentivos para convertir",
            "✓ Ofrecer descuentos limitados para crear urgencia",
            "✓ Mostrar valoraciones positivas de productos vistos",
            "✓ Considerar la activación de chat en vivo para resolver dudas",
            "✓ Implementar recordatorios de carrito si abandona sin comprar"
        ]
    else:
        recommendation["content"] = [
            "✓ Usuario probablemente en fase de exploración/investigación",
            "✓ Fomentar suscripción a newsletters o alertas de precio",
            "✓ Ofrecer contenido informativo sobre productos vistos",
            "✓ Implementar estrategias de remarketing a largo plazo",
            "✓ Mejorar la visualización de productos para aumentar interés"
        ]
    
    recommendations.append(recommendation)
    
    # Recomendación basada en PageValues
    if "PageValues" in input_data:
        page_rec = {"title": "", "content": []}
        
        if input_data["PageValues"] > 0:
            page_rec["title"] = "Optimización basada en PageValues"
            page_rec["content"] = [
                "✓ El usuario ha visitado páginas con alto potencial de conversión",
                "✓ Priorizar asistencia para este usuario (chatbot, soporte)",
                "✓ Ofrecer incentivos específicos para productos de interés",
                "✓ Considerar mensajes de escasez/urgencia si apropiado",
                "✓ Optimizar la ruta de conversión eliminando distracciones"
            ]
        else:
            page_rec["title"] = "Estrategia para aumentar PageValues"
            page_rec["content"] = [
                "✓ Mejorar la visibilidad de productos destacados/promociones",
                "✓ Implementar sistema de recomendación más efectivo",
                "✓ Revisar la UX para facilitar descubrimiento de productos",
                "✓ Personalizar contenido basado en comportamiento previo",
                "✓ Considerar rediseño de páginas clave para aumentar valor"
            ]
        
        recommendations.append(page_rec)
    
    # Recomendación basada en el mes
    if "Month" in input_data:
        month = input_data["Month"]
        month_rec = {"title": "", "content": []}
        
        # Meses de alta conversión
        if month in ["Nov", "Oct", "Sep"]:
            month_rec["title"] = f"Estrategia para temporada alta ({month})"
            month_rec["content"] = [
                "✓ Aprovechar la mayor propensión a comprar en esta temporada",
                "✓ Implementar ofertas especiales de temporada",
                "✓ Destacar productos populares o tendencia",
                "✓ Reforzar garantías de entrega para fechas clave",
                "✓ Considerar promociones con límite de tiempo"
            ]
        # Meses de baja conversión
        elif month in ["Jan", "Feb", "Apr"]:
            month_rec["title"] = f"Estrategia para temporada baja ({month})"
            month_rec["content"] = [
                "✓ Ofrecer incentivos especiales para compensar la menor propensión a comprar",
                "✓ Implementar estrategias de precio más agresivas",
                "✓ Enfatizar el valor a largo plazo de los productos",
                "✓ Considerar campañas temáticas especiales para estos meses",
                "✓ Invertir en adquisición de tráfico cualificado"
            ]
        else:
            month_rec["title"] = f"Estrategia estacional para {month}"
            month_rec["content"] = [
                "✓ Adaptar las ofertas según la estacionalidad del mes",
                "✓ Implementar campañas específicas para este período",
                "✓ Analizar datos históricos para optimizar conversiones",
                "✓ Segmentar mensajes según comportamiento estacional",
                "✓ Equilibrar la inversión publicitaria con la temporalidad"
            ]
        
        recommendations.append(month_rec)
    
    # Recomendación basada en tipo de visitante
    if "VisitorType" in input_data:
        visitor_rec = {"title": "", "content": []}
        
        if input_data["VisitorType"] == "New_Visitor":
            visitor_rec["title"] = "Estrategia para nuevos visitantes"
            visitor_rec["content"] = [
                "✓ Los nuevos visitantes tienen tasas de conversión de 24.9% vs 14.1% para recurrentes",
                "✓ Ofrecer incentivos especiales para primera compra",
                "✓ Simplificar el proceso de registro/checkout",
                "✓ Destacar propuestas de valor únicas",
                "✓ Implementar guías de navegación sencillas"
            ]
        else:
            visitor_rec["title"] = "Estrategia para visitantes recurrentes"
            visitor_rec["content"] = [
                "✓ Optimizar para aumentar la tasa de conversión (actualmente 14.1%)",
                "✓ Mostrar productos relacionados con historial previo",
                "✓ Implementar programas de fidelización",
                "✓ Ofrecer beneficios por compras recurrentes",
                "✓ Personalizar la experiencia basada en comportamientos anteriores"
            ]
        
        recommendations.append(visitor_rec)
    
    # Recomendación basada en patrones de navegación
    total_pages = input_data.get('Administrative', 0) + input_data.get('Informational', 0) + input_data.get('ProductRelated', 0)
    total_duration = input_data.get('Administrative_Duration', 0) + input_data.get('Informational_Duration', 0) + input_data.get('ProductRelated_Duration', 0)
    
    navigation_rec = {"title": "", "content": []}
    
    if total_pages > 20 and total_duration > 1000:
        navigation_rec["title"] = "Usuario con alto engagement"
        navigation_rec["content"] = [
            "✓ Patrón: muchas páginas, mucho tiempo (23.4% conversión)",
            "✓ Ofrecer asistencia personalizada basada en su navegación extensa",
            "✓ Proporcionar incentivos para finalizar la compra",
            "✓ Guardar sesión para continuar en otro momento",
            "✓ Implementar recordatorios si abandona sin comprar"
        ]
    elif total_pages <= 20 and total_duration > 1000:
        navigation_rec["title"] = "Usuario con navegación profunda"
        navigation_rec["content"] = [
            "✓ Patrón: pocas páginas, mucho tiempo (17.7% conversión)",
            "✓ El usuario muestra interés concentrado en pocos productos",
            "✓ Ofrecer información detallada y comparativas",
            "✓ Destacar recomendaciones complementarias relevantes",
            "✓ Facilitar reviews y testimonios específicos"
        ]
    elif total_pages > 20 and total_duration <= 1000:
        navigation_rec["title"] = "Usuario con navegación rápida"
        navigation_rec["content"] = [
            "✓ Patrón: muchas páginas, poco tiempo (12.5% conversión)",
            "✓ Mejorar herramientas de filtrado y comparación",
            "✓ Simplificar la presentación de información clave",
            "✓ Destacar bestsellers y productos mejor valorados",
            "✓ Implementar vistas rápidas sin cambio de página"
        ]
    else:
        navigation_rec["title"] = "Usuario con bajo engagement"
        navigation_rec["content"] = [
            "✓ Patrón: pocas páginas, poco tiempo (8.0% conversión)",
            "✓ Mejorar el primer impacto y tiempo de carga",
            "✓ Implementar banners atractivos y destacados",
            "✓ Simplificar la UX para fomentar más navegación",
            "✓ Considerar popups inteligentes antes de salida"
        ]
    
    recommendations.append(navigation_rec)
    
    return recommendations

# Sidebar para navegación
with st.sidebar:
    st.title("🛒 Navegación")
    page = st.radio("Selecciona una sección:", ["Historial de Predicciones", "Simulador"])
    
    st.markdown("---")
    st.markdown("### Sobre esta aplicación")
    
    # Mostrar el tipo de modelo correcto
    model_type = "XGBoost" if "XGBoost" in str(type(model)) else "Random Forest" if "RandomForest" in str(type(model)) else "Machine Learning"
    
    # Descripción del negocio
    st.markdown(f"""
    ### Optimización de Conversiones en E-Commerce
    
    Esta herramienta utiliza un modelo de **{model_type}** para predecir la probabilidad 
    de que un usuario realice una compra basándose en su comportamiento de navegación.
    
    **Aplicaciones de negocio:**
    
    - 📈 **Personalización en tiempo real** de la experiencia del usuario
    - 🎯 **Optimización de campañas** en periodos de alta conversión
    - 🛍️ **Mejora del diseño del sitio** para reducir abandonos
    - 📱 **Estrategias de retargeting** más efectivas
    - 💰 **Asignación eficiente** de recursos de marketing
    
    La tasa actual de conversión del sitio es del **15.6%**, con grandes 
    variaciones estacionales y diferencias entre tipos de visitantes.
    """)

# Página de historial de predicciones
if page == "Historial de Predicciones":
    st.title("📝 Historial de Predicciones")
    
    # Botón para recargar
    if st.button("🔄 Recargar historial"):
        st.experimental_rerun()
    
    # Cargar el historial de predicciones
    predictions_df = load_predictions()
    
    if predictions_df.empty:
        st.info("No hay predicciones guardadas aún. Usa el simulador para hacer predicciones.")
    else:
        # Mostrar estadísticas básicas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_predictions = len(predictions_df)
            st.metric("Total de predicciones", total_predictions)
        
        with col2:
            purchase_count = len(predictions_df[predictions_df['Prediction'] == 'Compra'])
            purchase_percent = (purchase_count / total_predictions) * 100 if total_predictions > 0 else 0
            st.metric("Predicciones de compra", f"{purchase_count} ({purchase_percent:.1f}%)")
        
        with col3:
            avg_probability = predictions_df['Probability'].str.rstrip('%').astype(float).mean() / 100
            st.metric("Probabilidad media", f"{avg_probability:.2%}")
        
        # Filtros para el historial
        st.subheader("Filtrar predicciones")
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_filter = st.multiselect(
                "Filtrar por resultado",
                options=["Compra", "No Compra"],
                default=["Compra", "No Compra"]
            )
        
        with col2:
            if 'Month' in predictions_df.columns:
                month_filter = st.multiselect(
                    "Filtrar por mes",
                    options=sorted(predictions_df['Month'].unique()),
                    default=sorted(predictions_df['Month'].unique())
                )
            else:
                month_filter = []
        
        # Aplicar filtros
        filtered_df = predictions_df
        if prediction_filter:
            filtered_df = filtered_df[filtered_df['Prediction'].isin(prediction_filter)]
        
        if month_filter and 'Month' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Month'].isin(month_filter)]
        
        # Mostrar el historial filtrado
        st.subheader("Resultados filtrados")
        st.dataframe(filtered_df)
        
        # Opción para descargar el historial
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar historial filtrado",
            data=csv,
            file_name="predicciones_filtradas.csv",
            mime="text/csv"
        )
        
        # Visualizaciones
        st.subheader("Visualizaciones")
        
        tab1, tab2 = st.tabs(["Distribución de predicciones", "Tendencias temporales"])
        
        with tab1:
            # Gráfico de distribución de predicciones
            fig, ax = plt.subplots(figsize=(10, 6))
            prediction_counts = filtered_df['Prediction'].value_counts()
            sns.barplot(x=prediction_counts.index, y=prediction_counts.values, ax=ax)
            ax.set_title("Distribución de predicciones")
            ax.set_ylabel("Cantidad")
            ax.set_xlabel("Predicción")
            st.pyplot(fig)
        
        with tab2:
            # Si hay suficientes datos, mostrar tendencia temporal
            if 'Timestamp' in filtered_df.columns and len(filtered_df) > 1:
                filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'])
                filtered_df['Date'] = filtered_df['Timestamp'].dt.date
                
                # Agrupar por fecha y contar predicciones
                daily_counts = filtered_df.groupby(['Date', 'Prediction']).size().unstack(fill_value=0)
                
                # Gráfico de tendencia temporal
                fig, ax = plt.subplots(figsize=(12, 6))
                daily_counts.plot(ax=ax)
                ax.set_title("Tendencia de predicciones a lo largo del tiempo")
                ax.set_ylabel("Cantidad de predicciones")
                ax.set_xlabel("Fecha")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Se necesitan más datos para mostrar tendencias temporales.")

# Página de simulador
elif page == "Simulador":
    st.title("🧪 Simulador de Comportamiento de Usuario")
    
    st.markdown("""
    Este simulador te permite experimentar con diferentes escenarios de comportamiento de usuario 
    para predecir la probabilidad de compra y obtener recomendaciones estratégicas de marketing.
    """)
    
    # Escenario base
    base_data = {
        'PageValues': 0.0,
        'ExitRates': 0.05,
        'BounceRates': 0.05,
        'Weekend': 0,
        'Administrative': 2,
        'Informational': 1,
        'ProductRelated': 10,
        'Administrative_Duration': 60.0,
        'Informational_Duration': 30.0,
        'ProductRelated_Duration': 180.0,
        'Month': 'Nov',
        'VisitorType': 'Returning_Visitor'
    }
    
    # Crear el simulador con sliders y controles
    st.subheader("Configura el escenario de usuario")
    
    # Dividir en dos columnas
    col1, col2 = st.columns(2)
    
    # Valores iniciales para simulación
    sim_data = {}
    
    with col1:
        # Principales indicadores de conversión
        st.markdown("#### Indicadores principales")
        
        sim_data['PageValues'] = st.slider(
            "PageValues (valor comercial de página)",
            min_value=0.0,
            max_value=50.0,
            value=base_data['PageValues'],
            step=0.1,
            help="El valor asignado a páginas visitadas (mayor valor indica mayor intención de compra)"
        )
        
        sim_data['ExitRates'] = st.slider(
            "ExitRates (tasa de salida)",
            min_value=0.0,
            max_value=0.2,
            value=base_data['ExitRates'],
            step=0.01,
            help="Porcentaje de salidas desde una página específica"
        )
        
        sim_data['BounceRates'] = st.slider(
            "BounceRates (tasa de rebote)",
            min_value=0.0,
            max_value=0.2,
            value=base_data['BounceRates'],
            step=0.01,
            help="Porcentaje de visitantes que abandonan el sitio tras ver una sola página"
        )
        
        # Comportamiento de páginas
        st.markdown("#### Páginas visitadas")
        
        sim_data['Administrative'] = st.slider(
            "Páginas administrativas",
            min_value=0,
            max_value=20,
            value=base_data['Administrative'],
            help="Número de páginas administrativas visitadas (carrito, cuenta, etc.)"
        )
        
        sim_data['Informational'] = st.slider(
            "Páginas informativas",
            min_value=0,
            max_value=20,
            value=base_data['Informational'],
            help="Número de páginas informativas visitadas (FAQ, políticas, etc.)"
        )
        
        sim_data['ProductRelated'] = st.slider(
            "Páginas de productos",
            min_value=0,
            max_value=100,
            value=base_data['ProductRelated'],
            help="Número de páginas de productos visitadas"
        )
    
    with col2:
        # Tiempo de navegación
        st.markdown("#### Tiempo de navegación (segundos)")
        
        sim_data['Administrative_Duration'] = st.slider(
            "Tiempo en páginas administrativas",
            min_value=0.0,
            max_value=500.0,
            value=base_data['Administrative_Duration'],
            step=10.0,
            help="Tiempo dedicado a páginas administrativas en segundos"
        )
        
        sim_data['Informational_Duration'] = st.slider(
            "Tiempo en páginas informativas",
            min_value=0.0,
            max_value=500.0,
            value=base_data['Informational_Duration'],
            step=10.0,
            help="Tiempo dedicado a páginas informativas en segundos"
        )
        
        sim_data['ProductRelated_Duration'] = st.slider(
            "Tiempo en páginas de productos",
            min_value=0.0,
            max_value=1000.0,
            value=base_data['ProductRelated_Duration'],
            step=10.0,
            help="Tiempo dedicado a páginas de productos en segundos"
        )
        
        # Contexto de visita
        st.markdown("#### Contexto de la visita")
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_index = months.index(base_data['Month']) if base_data['Month'] in months else 0
        
        sim_data['Month'] = st.selectbox(
            "Mes de la visita",
            options=months,
            index=month_index,
            help="Los meses con mayor conversión son: Nov (25.5%), Oct (20.9%), Sep (19.2%)"
        )
        
        visitor_types = ['Returning_Visitor', 'New_Visitor']
        visitor_index = visitor_types.index(base_data['VisitorType']) if base_data['VisitorType'] in visitor_types else 0
        
        sim_data['VisitorType'] = st.selectbox(
            "Tipo de visitante",
            options=visitor_types,
            index=visitor_index,
            help="Los nuevos visitantes tienen mayor tasa de conversión (24.9%) que los recurrentes (14.1%)"
        )
        
        sim_data['Weekend'] = 1 if st.checkbox(
            "¿Es fin de semana?",
            value=bool(base_data['Weekend']),
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
        if total_pages > 20 and total_duration > 1000:
            pattern = "Alto engagement (23.4% conv.)"
        elif total_pages <= 20 and total_duration > 1000:
            pattern = "Navegación profunda (17.7% conv.)"
        elif total_pages > 20 and total_duration <= 1000:
            pattern = "Navegación rápida (12.5% conv.)"
        else:
            pattern = "Bajo engagement (8.0% conv.)"
        st.metric("Patrón de navegación", pattern)
    
    # Botón para ejecutar simulación
    if st.button("🔮 Ejecutar simulación", use_container_width=True):
        with st.spinner("Analizando comportamiento de usuario..."):
            # Preprocesar los datos
            processed_data = preprocess_input_data(sim_data)
            
            # Realizar la predicción
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]
            
            # Mostrar el resultado
            st.subheader("Resultado de la predicción")
            
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.success("✅ PREDICCIÓN: USUARIO REALIZARÁ UNA COMPRA")
                else:
                    st.error("❌ PREDICCIÓN: USUARIO NO REALIZARÁ UNA COMPRA")
            
            with col2:
                st.metric("Probabilidad de compra", f"{probability:.2%}")
            
            # Mostrar barra de progreso
            st.progress(float(probability))
            
            # Generar recomendaciones de marketing
            recommendations = generate_marketing_recommendations(sim_data, probability)
            
            # Mostrar recomendaciones
            st.subheader("💡 Recomendaciones estratégicas de marketing")
            
            tabs = st.tabs([rec["title"] for rec in recommendations])
            
            for i, tab in enumerate(tabs):
                with tab:
                    for item in recommendations[i]["content"]:
                        st.markdown(item)
            
            # Mostrar características más influyentes
            if feature_importance is not None:
                st.subheader("🔍 Factores determinantes en la predicción")
                
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
                st.success("✅ Predicción guardada automáticamente en el historial")
            else:
                st.warning("⚠️ No se pudo guardar la predicción. Verifica los permisos de escritura")
            
            # Mostrar datos procesados
            with st.expander("Ver datos procesados"):
                st.subheader("Datos después del preprocesamiento")
                st.dataframe(processed_data)