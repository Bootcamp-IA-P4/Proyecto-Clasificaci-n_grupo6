import os

# Rutas de archivos relativas a la raíz del proyecto
MODEL_PATH = 'notebooks/modeling/xgboost/pkl_exports/xgboost_optimized.pkl'
PREPROCESSOR_PATH = 'notebooks/modeling/xgboost/pkl_exports/xgboost_preprocessor.pkl'
PREDICTIONS_PATH = 'data/predictions.csv'

# Configuración de la página
PAGE_CONFIG = {
    "page_title": "Predictor de Compras Online",
    "page_icon": "🛒",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Esquema de colores
COLORS = {
    "primary": "#1a73e8",       
    "secondary": "#5F6368",     
    "success": "#34A853",       
    "danger": "#EA4335",        
    "warning": "#FBBC05",      
    "background": "#FFFFFF",    
    "text": "#202124",          
    "chart_blue": "#4285F4",    
    "chart_blue_light": "#8ab4f8",  
    "light_gray": "#F1F3F4",   
    "border": "#DADCE0",        
    "orange": "#F9AB00",        
    "teal": "#00A9A7",          
    "purple": "#A142F4",        
    "sidebar_active": "#4285F4", 
    "sidebar_hover": "#E8F0FE",  
    "sidebar_inactive": "#F1F3F4" 
}

# Información general de la aplicación
APP_INFO = {
    "name": "Predictor de Compras Online",
    "description": "Esta herramienta utiliza un modelo de ML para predecir la probabilidad de que un usuario realice una compra basándose en su comportamiento de navegación.",
    "conversion_rate": "15.6%",
    "model_type": "XGBoost"
}

# Valores iniciales para simulación
DEFAULT_USER_DATA = {
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

# Opciones para selectores
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
VISITOR_TYPES = ['Returning_Visitor', 'New_Visitor', 'Other']