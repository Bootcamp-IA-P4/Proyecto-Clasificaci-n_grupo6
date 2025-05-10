import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from config import MODEL_PATH, PREPROCESSOR_PATH

@st.cache_resource
def load_model_and_preprocessor():
    """
    Carga el modelo y el preprocesador desde los archivos guardados.
    Utiliza cache_resource para evitar recargas innecesarias.
    """
    try:
        # Cargar el modelo
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            
            # Obtener características esperadas
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
            else:
                expected_features = []
        else:
            return None, None, [], None
        
        # Cargar el preprocesador
        preprocessor = None
        if os.path.exists(PREPROCESSOR_PATH):
            with open(PREPROCESSOR_PATH, 'rb') as f:
                preprocessor = pickle.load(f)
        
        # Obtener la importancia de características
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': expected_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
        return model, preprocessor, expected_features, feature_importance
    
    except Exception as e:
        return None, None, [], None

def preprocess_data(input_data, preprocessor, expected_features):
    """
    Preprocesa los datos de entrada utilizando la información del preprocesador.
    
    Args:
        input_data (dict): Datos de entrada del usuario
        preprocessor (dict): Preprocesador cargado
        expected_features (list): Lista de características que espera el modelo
        
    Returns:
        DataFrame: Datos preprocesados listos para el modelo
    """
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
        _manual_preprocessing(input_df)
    
    # Seleccionar solo las características que el modelo espera
    final_df = pd.DataFrame()
    for feature in expected_features:
        if feature in input_df.columns:
            final_df[feature] = input_df[feature]
        else:
            final_df[feature] = 0
    
    return final_df

def _manual_preprocessing(input_df):
    """
    Implementa el preprocesamiento manual si no hay preprocesador disponible.
    
    Args:
        input_df (DataFrame): DataFrame con los datos de entrada
    """
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

def make_prediction(model, processed_data):
    """
    Realiza una predicción con el modelo.
    
    Args:
        model: Modelo cargado
        processed_data (DataFrame): Datos preprocesados
        
    Returns:
        tuple: (predicción, probabilidad)
    """
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]
    
    return prediction, probability