import pytest
from config import MODEL_PATH, PREPROCESSOR_PATH
from app.model_utils import load_model_and_preprocessor, preprocess_data, make_prediction
from app.db_manager import save_prediction, load_predictions, get_prediction_stats
from app.db_models import Prediction

def test_save_and_load_prediction(db_session, sample_data):
    """Test de integración para guardar y cargar predicciones"""
    # Cargar modelo y preprocesador
    model, preprocessor, expected_features, _ = load_model_and_preprocessor()
    
    # Preprocesar datos
    processed_data = preprocess_data(sample_data, preprocessor, expected_features)
    
    # Generar predicción
    prediction, probability = make_prediction(model, processed_data)
    
    # Guardar predicción en base de datos
    save_result = save_prediction(sample_data, prediction, probability)
    assert save_result, "Fallo al guardar la predicción"
    
    # Cargar predicciones
    loaded_predictions = load_predictions()
    assert not loaded_predictions.empty, "No se cargaron predicciones"
    
    # Verificar la última predicción guardada
    last_prediction = loaded_predictions.iloc[0]
    assert last_prediction['prediction'] == ('Compra' if prediction == 1 else 'No Compra'), "Discrepancia en la predicción guardada"
    assert abs(last_prediction['prediction_rate'] - probability) < 0.001, "Discrepancia en la probabilidad"

def test_prediction_process_end_to_end(db_session, sample_data):
    """Test de integración de todo el proceso de predicción"""
    # Cargar modelo y preprocesador
    model, preprocessor, expected_features, _ = load_model_and_preprocessor()
    
    # Preprocesar datos
    processed_data = preprocess_data(sample_data, preprocessor, expected_features)
    
    # Generar predicción
    prediction, probability = make_prediction(model, processed_data)
    
    # Guardar predicción
    save_result = save_prediction(sample_data, prediction, probability)
    assert save_result, "Fallo al guardar la predicción"
    
    # Obtener estadísticas
    stats = get_prediction_stats()
    assert stats['total'] > 0, "No se guardaron predicciones"
    assert 0 <= stats['purchase_percent'] <= 100, "Porcentaje de compra inválido"
    assert 0 <= stats['avg_probability'] <= 1, "Probabilidad promedio inválida"