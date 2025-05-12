import pytest
import numpy as np
from config import MODEL_PATH, PREPROCESSOR_PATH
from model_utils import load_model_and_preprocessor, preprocess_data, make_prediction

#TESTS UNITARIOS
def test_load_model_and_preprocessor():
    """Test para verificar la carga del modelo y preprocesador"""
    model, preprocessor, expected_features, feature_importance = load_model_and_preprocessor()
    
    # Verificar que se cargaron correctamente
    assert model is not None, "El modelo no se cargó correctamente"
    assert preprocessor is not None, "El preprocesador no se cargó correctamente"
    assert len(expected_features) > 0, "No se encontraron características del modelo"
    assert feature_importance is not None, "La importancia de características no se cargó"

def test_model_prediction(modelo_base, preprocessor, sample_data):
    """Test para verificar las predicciones del modelo XGBoost"""
    # Obtener características esperadas y preprocesar
    _, _, expected_features, _ = load_model_and_preprocessor()
    processed_data = preprocess_data(sample_data, preprocessor, expected_features)
    
    # Realizar predicción
    prediction, probability = make_prediction(modelo_base, processed_data)
    
    # Verificaciones
    assert prediction in [0, 1], "La predicción debe ser 0 o 1"
    assert 0 <= probability <= 1, "La probabilidad debe estar entre 0 y 1"
    
    # Test adicional de consistencia de predicciones
    # Hacer múltiples predicciones con el mismo input para verificar consistencia
    predictions = [make_prediction(modelo_base, processed_data)[0] for _ in range(5)]
    assert all(p == predictions[0] for p in predictions), "Las predicciones no son consistentes"

def test_preprocess_data(preprocessor, sample_data):
    """Test para verificar el preprocesamiento de datos"""
    # Obtener características esperadas
    _, _, expected_features, _ = load_model_and_preprocessor()
    
    # Preprocesar datos
    processed_data = preprocess_data(sample_data, preprocessor, expected_features)
    
    # Verificaciones
    assert processed_data is not None, "El preprocesamiento falló"
    assert processed_data.shape[1] == len(expected_features), "Número de características incorrecto"
    assert all(feature in processed_data.columns for feature in expected_features), "Falta alguna característica esperada"