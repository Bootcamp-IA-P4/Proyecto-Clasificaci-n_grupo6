import pytest
import sys
import os

# Obtener la ruta absoluta del directorio raíz del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Importaciones necesarias para los tests
from model_utils import load_model_and_preprocessor, preprocess_data, make_prediction
from db_manager import get_db, save_prediction, create_tables
from config import MODEL_PATH, PREPROCESSOR_PATH  
import pandas as pd
import pickle

@pytest.fixture(scope="session")
def modelo_base():
    """Fixture para cargar el modelo base"""
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@pytest.fixture(scope="session")
def preprocessor():
    """Fixture para cargar el preprocesador"""
    with open(PREPROCESSOR_PATH, 'rb') as f:
        return pickle.load(f)

@pytest.fixture
def sample_data():
    """Fixture con un ejemplo de datos de entrada"""
    return {
        'PageValues': 50.0,
        'ExitRates': 0.2,
        'BounceRates': 0.3,
        'Weekend': 0,
        'Administrative': 1,
        'Informational': 2,
        'ProductRelated': 3,
        'Administrative_Duration': 50,
        'Informational_Duration': 100,
        'ProductRelated_Duration': 150,
        'Month': 'May',
        'VisitorType': 'Returning_Visitor'
    }

@pytest.fixture(scope="function")
def db_session():
    """Fixture para obtener una sesión de base de datos"""
    # Crear tablas antes de cada test
    create_tables()
    
    # Obtener sesión
    db = get_db()
    try:
        yield db
    finally:
        # Cerrar sesión después de cada test
        db.close()