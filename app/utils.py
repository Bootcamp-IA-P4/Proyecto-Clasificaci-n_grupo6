import streamlit as st
import pandas as pd
import numpy as np
from config import COLORS

def get_navigation_pattern(total_pages, total_duration):
    """
    Determina el patrón de navegación basado en el total de páginas
    y la duración total de la sesión.
    
    Args:
        total_pages (int): Número total de páginas visitadas
        total_duration (float): Duración total en segundos
        
    Returns:
        tuple: (pattern_name, description, conversion_rate)
    """
    if total_pages > 20 and total_duration > 1000:
        return (
            "Alto engagement",
            "Muchas páginas, mucho tiempo",
            "23.4% conversión"
        )
    elif total_pages <= 20 and total_duration > 1000:
        return (
            "Navegación profunda",
            "Pocas páginas, mucho tiempo",
            "17.7% conversión"
        )
    elif total_pages > 20 and total_duration <= 1000:
        return (
            "Navegación rápida",
            "Muchas páginas, poco tiempo",
            "12.5% conversión"
        )
    else:
        return (
            "Bajo engagement",
            "Pocas páginas, poco tiempo",
            "8.0% conversión"
        )

def generate_marketing_recommendations(input_data, probability):
    """
    Genera recomendaciones de marketing basadas en los datos de entrada y la probabilidad.
    
    Args:
        input_data (dict): Datos de entrada del usuario
        probability (float): Probabilidad de compra
        
    Returns:
        list: Lista de diccionarios con recomendaciones
    """
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
            "Este usuario muestra una alta intención de compra",
            "Maximizar el valor del carrito con recomendaciones complementarias",
            "Ofrecer opciones de envío premium o garantías extendidas",
            "Mostrar productos similares de mayor categoría/precio",
            "Minimizar distracciones en el proceso de checkout"
        ]
    elif probability_level == "media":
        recommendation["content"] = [
            "El usuario muestra interés pero necesita incentivos para convertir",
            "Ofrecer descuentos limitados para crear urgencia",
            "Mostrar valoraciones positivas de productos vistos",
            "Considerar la activación de chat en vivo para resolver dudas",
            "Implementar recordatorios de carrito si abandona sin comprar"
        ]
    else:
        recommendation["content"] = [
            "Usuario probablemente en fase de exploración/investigación",
            "Fomentar suscripción a newsletters o alertas de precio",
            "Ofrecer contenido informativo sobre productos vistos",
            "Implementar estrategias de remarketing a largo plazo",
            "Mejorar la visualización de productos para aumentar interés"
        ]
    
    recommendations.append(recommendation)
    
    # Recomendación basada en PageValues
    if "PageValues" in input_data:
        page_rec = {"title": "", "content": []}
        
        if input_data["PageValues"] > 0:
            page_rec["title"] = "Optimización basada en PageValues"
            page_rec["content"] = [
                "El usuario ha visitado páginas con alto potencial de conversión",
                "Priorizar asistencia para este usuario (chatbot, soporte)",
                "Ofrecer incentivos específicos para productos de interés",
                "Considerar mensajes de escasez/urgencia si apropiado",
                "Optimizar la ruta de conversión eliminando distracciones"
            ]
        else:
            page_rec["title"] = "Estrategia para aumentar PageValues"
            page_rec["content"] = [
                "Mejorar la visibilidad de productos destacados/promociones",
                "Implementar sistema de recomendación más efectivo",
                "Revisar la UX para facilitar descubrimiento de productos",
                "Personalizar contenido basado en comportamiento previo",
                "Considerar rediseño de páginas clave para aumentar valor"
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
                "Aprovechar la mayor propensión a comprar en esta temporada",
                "Implementar ofertas especiales de temporada",
                "Destacar productos populares o tendencia",
                "Reforzar garantías de entrega para fechas clave",
                "Considerar promociones con límite de tiempo"
            ]
        # Meses de baja conversión
        elif month in ["Jan", "Feb", "Apr"]:
            month_rec["title"] = f"Estrategia para temporada baja ({month})"
            month_rec["content"] = [
                "Ofrecer incentivos especiales para compensar la menor propensión a comprar",
                "Implementar estrategias de precio más agresivas",
                "Enfatizar el valor a largo plazo de los productos",
                "Considerar campañas temáticas especiales para estos meses",
                "Invertir en adquisición de tráfico cualificado"
            ]
        else:
            month_rec["title"] = f"Estrategia estacional para {month}"
            month_rec["content"] = [
                "Adaptar las ofertas según la estacionalidad del mes",
                "Implementar campañas específicas para este período",
                "Analizar datos históricos para optimizar conversiones",
                "Segmentar mensajes según comportamiento estacional",
                "Equilibrar la inversión publicitaria con la temporalidad"
            ]
        
        recommendations.append(month_rec)
    
    # Recomendación basada en tipo de visitante
    if "VisitorType" in input_data:
        visitor_rec = {"title": "", "content": []}
        
        if input_data["VisitorType"] == "New_Visitor":
            visitor_rec["title"] = "Estrategia para nuevos visitantes"
            visitor_rec["content"] = [
                "Los nuevos visitantes tienen tasas de conversión de 24.9% vs 14.1% para recurrentes",
                "Ofrecer incentivos especiales para primera compra",
                "Simplificar el proceso de registro/checkout",
                "Destacar propuestas de valor únicas",
                "Implementar guías de navegación sencillas"
            ]
        else:
            visitor_rec["title"] = "Estrategia para visitantes recurrentes"
            visitor_rec["content"] = [
                "Optimizar para aumentar la tasa de conversión (actualmente 14.1%)",
                "Mostrar productos relacionados con historial previo",
                "Implementar programas de fidelización",
                "Ofrecer beneficios por compras recurrentes",
                "Personalizar la experiencia basada en comportamientos anteriores"
            ]
        
        recommendations.append(visitor_rec)
    
    # Recomendación basada en patrones de navegación
    total_pages = input_data.get('Administrative', 0) + input_data.get('Informational', 0) + input_data.get('ProductRelated', 0)
    total_duration = input_data.get('Administrative_Duration', 0) + input_data.get('Informational_Duration', 0) + input_data.get('ProductRelated_Duration', 0)
    
    pattern_name, pattern_desc, pattern_conv = get_navigation_pattern(total_pages, total_duration)
    
    navigation_rec = {
        "title": f"Usuario con {pattern_name.lower()}",
        "content": [
            f"Patrón: {pattern_desc} ({pattern_conv})",
            get_navigation_recommendations(pattern_name)[0],
            get_navigation_recommendations(pattern_name)[1],
            get_navigation_recommendations(pattern_name)[2],
            get_navigation_recommendations(pattern_name)[3],
            get_navigation_recommendations(pattern_name)[4]
        ]
    }
    
    recommendations.append(navigation_rec)
    
    return recommendations

def get_navigation_recommendations(pattern):
    """
    Obtiene recomendaciones basadas en el patrón de navegación.
    
    Args:
        pattern (str): Nombre del patrón de navegación
        
    Returns:
        list: Lista con recomendaciones específicas
    """
    recommendations = {
        "Alto engagement": [
            "Ofrecer asistencia personalizada basada en su navegación extensa",
            "Proporcionar incentivos para finalizar la compra",
            "Guardar sesión para continuar en otro momento",
            "Implementar recordatorios si abandona sin comprar",
            "Sugerir productos complementarios basados en su navegación profunda"
        ],
        "Navegación profunda": [
            "El usuario muestra interés concentrado en pocos productos",
            "Ofrecer información detallada y comparativas",
            "Destacar recomendaciones complementarias relevantes",
            "Facilitar reviews y testimonios específicos",
            "Proporcionar garantías adicionales para productos de interés"
        ],
        "Navegación rápida": [
            "Mejorar herramientas de filtrado y comparación",
            "Simplificar la presentación de información clave",
            "Destacar bestsellers y productos mejor valorados",
            "Implementar vistas rápidas sin cambio de página",
            "Optimizar la búsqueda y categorización de productos"
        ],
        "Bajo engagement": [
            "Mejorar el primer impacto y tiempo de carga",
            "Implementar banners atractivos y destacados",
            "Simplificar la UX para fomentar más navegación",
            "Considerar popups inteligentes antes de salida",
            "Ofrecer descuentos por primera compra más visibles"
        ]
    }
    
    return recommendations.get(pattern, ["No hay recomendaciones específicas disponibles"] * 5)