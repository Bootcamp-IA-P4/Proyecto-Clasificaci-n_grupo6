import os
import csv
import pandas as pd
from datetime import datetime
from config import PREDICTIONS_PATH
from db_manager import save_prediction as db_save_prediction
from db_manager import load_predictions as db_load_predictions
from db_manager import filter_predictions as db_filter_predictions
from db_manager import get_prediction_stats as db_get_prediction_stats
from db_manager import get_temporal_data as db_get_temporal_data

def save_prediction(input_data, prediction, probability):
    """
    Guarda una predicción en la base de datos.
    
    Args:
        input_data (dict): Datos de entrada del usuario
        prediction (int): Predicción del modelo (1=Compra, 0=No Compra)
        probability (float): Probabilidad de compra
        
    Returns:
        bool: True si se guardó correctamente, False en caso contrario
    """
    # Llamamos a la implementación de la base de datos
    return db_save_prediction(input_data, prediction, probability)

def load_predictions():
    """
    Carga el historial de predicciones desde la base de datos.
    
    Returns:
        DataFrame: Historial de predicciones o DataFrame vacío si no existe
    """
    # Llamamos a la implementación de la base de datos
    return db_load_predictions()

def filter_predictions(df, filters=None):
    """
    Filtra el DataFrame de predicciones según los criterios especificados.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
        filters (dict): Diccionario con los filtros a aplicar
        
    Returns:
        DataFrame: DataFrame filtrado
    """
    # Si no hay filtros o el DataFrame está vacío, devolvemos el DataFrame original
    if filters is None or df.empty:
        return df
    
    # Llamamos a la implementación de la base de datos
    # Si se proporciona un DataFrame, lo usamos. De lo contrario, obtenemos datos de la BD
    if df is not None and not df.empty:
        # Compatibilidad con el código existente que usa DataFrames
        filtered_df = df.copy()
        
        # Aplicar filtros
        for column, values in filters.items():
            if column in filtered_df.columns and values:
                filtered_df = filtered_df[filtered_df[column].isin(values)]
        
        return filtered_df
    else:
        # Obtenemos datos directamente de la base de datos
        return db_filter_predictions(filters)

def get_prediction_stats(df):
    """
    Calcula estadísticas básicas sobre las predicciones.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
        
    Returns:
        dict: Diccionario con estadísticas
    """
    if df is None or df.empty:
        return {
            "total": 0,
            "purchases": 0,
            "purchase_percent": 0,
            "avg_probability": 0
        }
    
    # Calculamos estadísticas a partir del DataFrame proporcionado
    total = len(df)
    purchases = len(df[df['prediction'] == 'Compra'])
    purchase_percent = (purchases / total) * 100 if total > 0 else 0
    
    # Convertir string de porcentaje a float
    if 'probability' in df.columns:
        avg_probability = df['probability'].str.rstrip('%').astype(float).mean() / 100
    else:
        avg_probability = 0
    
    return {
        "total": total,
        "purchases": purchases,
        "purchase_percent": purchase_percent,
        "avg_probability": avg_probability
    }

def get_temporal_data(df):
    """
    Prepara datos para visualizaciones temporales.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
        
    Returns:
        DataFrame: DataFrame agrupado por fecha
    """
    if df is None or df.empty or 'timestamp' not in df.columns:
        return pd.DataFrame()
    
    # Convertir timestamp a datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Date'] = df['timestamp'].dt.date
    
    # Agrupar por fecha y contar predicciones
    daily_counts = df.groupby(['Date', 'prediction']).size().unstack(fill_value=0)
    
    return daily_counts