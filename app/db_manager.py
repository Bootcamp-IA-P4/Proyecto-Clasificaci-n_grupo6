import os
import logging
from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import pandas as pd
from db_models import Base, Prediction  # Cambiar de app.db_models a db_models

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Crear directorio data si no existe
DATA_PATH = os.getenv("DATA_PATH", "data/database")
os.makedirs(DATA_PATH, exist_ok=True)

# Configuración de la base de datos
DB_FILENAME = os.getenv("DB_FILENAME", "onlineshopping.db")
DB_URI = os.getenv("DB_URI", f"sqlite:///{DATA_PATH}/{DB_FILENAME}")

# Crear motor y sesión
engine = create_engine(DB_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Crea las tablas en la base de datos si no existen"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Tablas creadas correctamente")
    except Exception as e:
        logger.error(f"Error creando tablas: {e}")
        raise

def get_db():
    """Obtiene una sesión de base de datos"""
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        logger.error(f"Error al obtener conexión a la base de datos: {e}")
        raise

def save_prediction(input_data, prediction_value, probability):
    """
    Guarda una predicción en la base de datos
    
    Args:
        input_data (dict): Datos de entrada del usuario
        prediction_value (int): Predicción (1=Compra, 0=No Compra)
        probability (float): Probabilidad de compra
        
    Returns:
        bool: True si tuvo éxito, False si falló
    """
    try:
        db = get_db()
        
        # Crear objeto de predicción
        prediction = Prediction(
            prediction=bool(prediction_value),
            prediction_rate=probability,
            page_values=input_data.get('PageValues'),
            exit_rates=input_data.get('ExitRates'),
            bounce_rates=input_data.get('BounceRates'),
            weekend=bool(input_data.get('Weekend')),
            administrative=input_data.get('Administrative'),
            informational=input_data.get('Informational'),
            product_related=input_data.get('ProductRelated'),
            administrative_duration=input_data.get('Administrative_Duration'),
            informational_duration=input_data.get('Informational_Duration'),
            product_related_duration=input_data.get('ProductRelated_Duration'),
            month=input_data.get('Month'),
            visitor_type=input_data.get('VisitorType')
        )
        
        db.add(prediction)
        db.commit()
        db.close()
        return True
    except Exception as e:
        try:
            db.rollback()
        except:
            pass
        try:
            db.close()
        except:
            pass
        logger.error(f"Error guardando predicción: {e}")
        return False

def load_predictions():
    """
    Carga el historial de predicciones desde la base de datos
    
    Returns:
        DataFrame: Historial de predicciones o DataFrame vacío si no hay datos
    """
    try:
        db = get_db()
        predictions = db.query(Prediction).order_by(desc(Prediction.timestamp)).all()
        db.close()
        
        if not predictions:
            return pd.DataFrame()
        
        # Convertir a DataFrame
        data = [p.to_dict() for p in predictions]
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error cargando predicciones: {e}")
        return pd.DataFrame()

def filter_predictions(filters=None):
    """
    Filtra el DataFrame de predicciones según los criterios especificados
    
    Args:
        filters (dict): Diccionario con los filtros a aplicar
        
    Returns:
        DataFrame: DataFrame filtrado
    """
    try:
        db = get_db()
        query = db.query(Prediction)
        
        # Aplicar filtros si existen
        if filters:
            if 'start_date' in filters and filters['start_date']:
                query = query.filter(Prediction.timestamp >= filters['start_date'])
            
            if 'end_date' in filters and filters['end_date']:
                query = query.filter(Prediction.timestamp <= filters['end_date'])
            
            if 'prediction' in filters and filters['prediction']:
                pred_values = [p == "Compra" for p in filters['prediction']]
                query = query.filter(Prediction.prediction.in_(pred_values))
            
            if 'month' in filters and filters['month']:
                query = query.filter(Prediction.month.in_(filters['month']))
            
            if 'visitor_type' in filters and filters['visitor_type']:
                query = query.filter(Prediction.visitor_type.in_(filters['visitor_type']))
        
        # Ordenar por fecha descendente
        predictions = query.order_by(desc(Prediction.timestamp)).all()
        db.close()
        
        # Convertir a DataFrame
        data = [p.to_dict() for p in predictions]
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error filtrando predicciones: {e}")
        return pd.DataFrame()

def get_prediction_stats(df=None):
    """
    Calcula estadísticas sobre las predicciones
    
    Args:
        df (DataFrame, optional): DataFrame con las predicciones
        
    Returns:
        dict: Diccionario con estadísticas
    """
    if df is None:
        try:
            db = get_db()
            total = db.query(func.count(Prediction.id)).scalar() or 0
            purchases = db.query(func.count(Prediction.id)).filter(Prediction.prediction == True).scalar() or 0
            avg_probability = db.query(func.avg(Prediction.prediction_rate)).scalar() or 0
            db.close()
            
            purchase_percent = (purchases / total) * 100 if total > 0 else 0
            
            return {
                "total": total,
                "purchases": purchases,
                "purchase_percent": purchase_percent,
                "avg_probability": avg_probability
            }
        except Exception as e:
            logger.error(f"Error calculando estadísticas: {e}")
            return {
                "total": 0,
                "purchases": 0,
                "purchase_percent": 0,
                "avg_probability": 0
            }
    else:
        # Compatibilidad con el código existente
        if df.empty:
            return {
                "total": 0,
                "purchases": 0,
                "purchase_percent": 0,
                "avg_probability": 0
            }
        
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

def get_temporal_data(df=None):
    """
    Prepara datos para visualizaciones temporales
    
    Args:
        df (DataFrame, optional): DataFrame con las predicciones
        
    Returns:
        DataFrame: DataFrame agrupado por fecha
    """
    if df is None:
        df = load_predictions()
    
    if df.empty or 'timestamp' not in df.columns:
        return pd.DataFrame()
    
    # Convertir timestamp a datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Date'] = df['timestamp'].dt.date
    
    # Agrupar por fecha y contar predicciones
    daily_counts = df.groupby(['Date', 'prediction']).size().unstack(fill_value=0)
    
    return daily_counts

# Inicializar la base de datos automáticamente
create_tables()