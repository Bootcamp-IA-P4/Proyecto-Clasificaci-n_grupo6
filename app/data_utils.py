import os
import csv
import pandas as pd
from datetime import datetime
from config import PREDICTIONS_PATH

def save_prediction(input_data, prediction, probability):
    """
    Guarda una predicción en un archivo CSV.
    
    Args:
        input_data (dict): Datos de entrada del usuario
        prediction (int): Predicción del modelo (1=Compra, 0=No Compra)
        probability (float): Probabilidad de compra
        
    Returns:
        bool: True si se guardó correctamente, False en caso contrario
    """
    try:
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
        os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
        
        # Verificar si el archivo existe
        file_exists = os.path.isfile(PREDICTIONS_PATH)
        
        # Guardar al CSV
        if not file_exists:
            with open(PREDICTIONS_PATH, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=prediction_data.keys())
                writer.writeheader()
                writer.writerow(prediction_data)
        else:
            # Si existe, verificar columnas y añadir fila
            existing_df = pd.read_csv(PREDICTIONS_PATH)
            # Verificar si hay columnas nuevas
            missing_cols = set(prediction_data.keys()) - set(existing_df.columns)
            if missing_cols:
                # Añadir columnas nuevas a los datos existentes
                for col in missing_cols:
                    existing_df[col] = None
            # Añadir nueva fila y guardar
            new_row = pd.DataFrame([prediction_data])
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            updated_df.to_csv(PREDICTIONS_PATH, index=False)
        
        return True
    except Exception as e:
        print(f"Error al guardar la predicción: {str(e)}")
        return False

def load_predictions():
    """
    Carga el historial de predicciones desde el archivo CSV.
    
    Returns:
        DataFrame: Historial de predicciones o DataFrame vacío si no existe
    """
    if os.path.exists(PREDICTIONS_PATH):
        try:
            return pd.read_csv(PREDICTIONS_PATH)
        except Exception as e:
            print(f"Error al cargar el historial: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

def filter_predictions(df, filters=None):
    """
    Filtra el DataFrame de predicciones según los criterios especificados.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
        filters (dict): Diccionario con los filtros a aplicar
        
    Returns:
        DataFrame: DataFrame filtrado
    """
    if filters is None or df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Aplicar filtros
    for column, values in filters.items():
        if column in filtered_df.columns and values:
            filtered_df = filtered_df[filtered_df[column].isin(values)]
    
    return filtered_df

def get_prediction_stats(df):
    """
    Calcula estadísticas básicas sobre las predicciones.
    
    Args:
        df (DataFrame): DataFrame con las predicciones
        
    Returns:
        dict: Diccionario con estadísticas
    """
    if df.empty:
        return {
            "total": 0,
            "purchases": 0,
            "purchase_percent": 0,
            "avg_probability": 0
        }
    
    total = len(df)
    purchases = len(df[df['Prediction'] == 'Compra'])
    purchase_percent = (purchases / total) * 100 if total > 0 else 0
    
    # Convertir string de porcentaje a float
    if 'Probability' in df.columns:
        avg_probability = df['Probability'].str.rstrip('%').astype(float).mean() / 100
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
    if df.empty or 'Timestamp' not in df.columns:
        return pd.DataFrame()
    
    # Convertir timestamp a datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    
    # Agrupar por fecha y contar predicciones
    daily_counts = df.groupby(['Date', 'Prediction']).size().unstack(fill_value=0)
    
    return daily_counts