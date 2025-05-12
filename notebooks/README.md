# 📊 Análisis Exploratorio del Dataset de Intención de Compra 🛒

Este documento presenta el análisis exploratorio de datos (EDA) realizado sobre el dataset "Online Shoppers Purchasing Intention", que sirve como base para el desarrollo del modelo predictivo implementado en DataShop Analytics.

## 📋 Origen y Contexto del Dataset

El conjunto de datos proviene del [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) y contiene información sobre **12,330 sesiones de navegación** en un sitio de comercio electrónico. Cada registro representa una sesión de usuario, con variables que describen su comportamiento de navegación y si la sesión resultó en una compra.

**Características principales del dataset:**
- **Periodo de recolección**: Un año (datos reales de Google Analytics)
- **Estructura**: 18 variables (10 numéricas, 8 categóricas) incluyendo la variable objetivo
- **Variable objetivo**: `Revenue` (booleana, indica si la sesión resultó en una transacción)
- **Tasa de conversión**: 15.6% (común en e-commerce)

El dataset resulta particularmente valioso porque:
1. Contiene datos reales de comportamiento (no simulados)
2. Incluye variables clave de analítica web (tasas de rebote, valores de página, etc.)
3. Presenta un desbalance de clases típico en escenarios reales (pocas conversiones vs. muchos abandonos)

## 🎯 Objetivos del Análisis

El análisis exploratorio se diseñó para responder preguntas clave de negocio:

1. **¿Qué factores influyen más en la decisión de compra?**
2. **¿Existen patrones estacionales o temporales en las conversiones?**
3. **¿Qué segmentos de usuarios muestran mayor probabilidad de compra?**
4. **¿Cómo se relaciona el comportamiento de navegación con la conversión?**
5. **¿Qué variables deberían incluirse en un modelo predictivo eficaz?**

Estos insights son fundamentales para optimizar estrategias de marketing, diseño web y personalización de experiencia de usuario.

## 🔍 Proceso de Exploración

### 1. Análisis Descriptivo y Preparación

- **Verificación de calidad**: No se encontraron valores nulos, pero sí 125 registros duplicados (~1% del dataset).
- **Limpieza básica**: Eliminación de duplicados, resultando en 12,205 sesiones únicas.
- **Distribución objetivo**: Confirmación del desbalance - 15.6% de sesiones con conversión vs. 84.4% sin compra.

### 2. Análisis Univariado

- **Variables numéricas**: Distribuciones altamente sesgadas con concentración en valores bajos y outliers significativos.
- **Variables categóricas**: Patrones estacionales en `Month`, predominio de visitantes recurrentes (86%), y mayoría de sesiones en días laborables (77%).
- **Outliers**: Identificados como comportamientos reales y valiosos, no como errores de datos.

### 3. Análisis Bivariado y Correlaciones

- **Correlaciones con Revenue**: 
  - `PageValues` destaca significativamente (corr: 0.49)
  - `ExitRates` muestra correlación negativa (corr: -0.20)
  - `BounceRates` también negativa (corr: -0.15)
  - Variables de productos (`ProductRelated`, `ProductRelated_Duration`) con correlaciones moderadas (~0.15)

- **Multicolinealidad**:
  - Alta correlación entre `BounceRates` y `ExitRates` (corr: 0.90)
  - Fuerte correlación entre páginas visitadas y sus duraciones respectivas

### 4. Análisis Detallado de Variables Clave

- **PageValues**: 
  - Diferencia extraordinaria entre sesiones con/sin compra
  - Actúa casi como un predictor binario
  - Sesiones con `PageValues > 0` muestran alta probabilidad de conversión

- **Métricas de abandono**:
  - Tasas de rebote y salida significativamente menores en sesiones con compra
  - `ExitRates` cerca de cero como fuerte predictor de posible conversión

- **Comportamiento de navegación**:
  - Mayor número de páginas visitadas en sesiones con compra
  - Tiempo significativamente mayor en sesiones con conversión
  - Diferencias más pronunciadas en páginas de productos y administrativas

### 5. Análisis de Variables Categóricas

- **Estacionalidad marcada**:
  - Noviembre (25.5%), octubre (20.9%) y septiembre (19.2%) con tasas más altas
  - Enero (1.7%) con la tasa más baja - diferencia de 14x entre mejor y peor mes
  - Progresión clara: aumento gradual de julio a noviembre, caída en diciembre

- **Tipo de visitante**:
  - Nuevos visitantes con mayor tasa de conversión (24.9%) que recurrentes (14.1%)
  - Hallazgo contra-intuitivo que sugiere oportunidades de optimización

- **Fin de semana**:
  - Ligera ventaja para fines de semana (17.5% vs. 15.1% en días laborables)

### 6. Creación y Análisis de Variables Derivadas

- **Transformaciones logarítmicas**:
  - Mejoran distribuciones sesgadas de variables de duración
  - `ProductRelated_Duration_Log` con la correlación más alta (0.196)

- **Variables binarias**:
  - `PageValues_NonZero`: correlación más alta (0.602) con la variable objetivo
  - Confirma hallazgos sobre la importancia predictiva de `PageValues`

- **Variables agregadas**:
  - `TotalPages` y `TotalDuration` como métricas de engagement global
  - Correlaciones significativas con `Revenue` (0.162 y 0.154 respectivamente)

### 7. Análisis de Segmentos

- **Patrones de engagement**:
  - Alto engagement (páginas y tiempo): 23.4% de conversión (42.1% del total)
  - Bajo engagement: 8.1% de conversión (42.4% del total)

- **Combinación con PageValues**:
  - Bajo engagement + PageValues > 0: 82.4% de conversión (segmento pequeño: 2.9%)
  - Alto engagement + PageValues > 0: 49.1% de conversión (segmento significativo: 16.8%)
  - Alto engagement + PageValues = 0: solo 6.5% de conversión (25.4% del total)

## 📈 Principales Hallazgos

1. **PageValues como predictor dominante**:
   - Correlación extraordinaria con conversiones (0.49)
   - Como variable binaria (>0), alcanza correlación de 0.602
   - Incluso usuarios con bajo engagement pero PageValues>0 tienen alta probabilidad de compra (82.4%)

2. **Estacionalidad pronunciada**:
   - Trimestre septiembre-noviembre con tasas 5-10 veces superiores a principios de año
   - Oportunidad de estrategias temporales específicas

3. **Comportamiento de navegación diferenciado**:
   - Compradores visitan más páginas (mediana: 32 vs. 18)
   - Pasan más tiempo en el sitio (mediana: 1252 vs. 600 segundos)
   - Exploran más productos (mediana: 29 vs. 16)

4. **Menor abandono en compradores**:
   - ExitRates más bajos (0.020 vs. 0.046)
   - BounceRates cercanos a cero (0.005 vs. 0.023)

5. **Nuevos visitantes con mayor conversión**:
   - Tasa de 24.9% vs. 14.1% para visitantes recurrentes
   - Sugiere potencial de optimización para usuarios recurrentes

## 🧪 Variables Seleccionadas para Modelado

Basándonos en el análisis, recomendamos las siguientes variables para el modelo predictivo:

### Variables Originales Clave:
- **`PageValues`**: Principal predictor numérico
- **`ExitRates`**: Mejor indicador de abandono
- **`Weekend`**: Captura diferencias entre días laborables y fines de semana

### Variables Transformadas:
- **`Administrative_Duration_Log`**, **`Informational_Duration_Log`**, **`ProductRelated_Duration_Log`**: Transformaciones logarítmicas

### Variables Derivadas:
- **`PageValues_NonZero`**: Indicador binario (PageValues > 0)
- **`TotalPages`**: Número total de páginas visitadas
- **`TotalDuration`**: Tiempo total en el sitio

### Variables Categóricas Codificadas:
- **`Month`**: Variables dummy para todos los meses
- **`VisitorType`**: Categorías principales

## 🔮 Recomendaciones para el Modelado

### Técnicas de Preprocesamiento:
- **Transformaciones logarítmicas** para variables de duración
- **Creación de variables binarias derivadas** (especialmente PageValues_NonZero)
- **Codificación one-hot** para variables categóricas
- **Manejo del desbalance de clases** mediante técnicas como SMOTE

### Algoritmos Sugeridos:
1. **Random Forest**: Captura bien relaciones no lineales, robusto a outliers
2. **XGBoost**: Rendimiento superior, ideal para datos con clases desbalanceadas
3. **Regresión Logística**: Para interpretabilidad clara
4. **SVM**: Con kernel RBF para capturar relaciones no lineales

### Métricas de Evaluación:
- Priorizar **F1-score** dado el desbalance de clases
- Monitorear **precision** y **recall** para diferentes aspectos del rendimiento

## 🚀 Implicaciones para el Negocio

Este análisis proporciona insights accionables para optimizar estrategias de e-commerce:

1. **Personalización basada en segmentos**:
   - Identificar y enfocar esfuerzos en usuarios de alto valor
   - Adaptar experiencia según patrones de navegación

2. **Optimización estacional**:
   - Concentrar presupuesto publicitario en trimestre septiembre-noviembre
   - Implementar promociones especiales en meses de baja conversión

3. **Mejora de experiencia**:
   - Reducir tasas de rebote y salida
   - Optimizar páginas de productos y administrativas
   - Mejorar experiencia para visitantes recurrentes

4. **Estrategias de marketing dirigidas**:
   - Personalizar mensajes para nuevos vs. visitantes recurrentes
   - Ofrecer asistencia en tiempo real para usuarios con alta probabilidad de compra

---

El modelo de machine learning implementado en DataShop Analytics se ha desarrollado siguiendo estas recomendaciones, seleccionando XGBoost como algoritmo final por su rendimiento superior en métricas clave.