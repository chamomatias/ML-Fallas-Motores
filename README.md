# Predicción de fallas en motores industriales con Machine Learning

Este proyecto implementa un modelo de clasificación supervisada para predecir fallas en motores industriales con al menos 15 días de anticipación, a partir de variables sensoriales y operativas.

## Objetivo

Desarrollar una solución predictiva capaz de anticipar si un motor presentará una falla en los próximos 15 días, con base en datos históricos como temperatura, vibración, presión, RPM, consumo energético y horas de operación.

## Dataset

El conjunto de datos contiene más de 63.000 registros correspondientes a 50 motores distintos, cada uno con mediciones diarias. Las variables incluyen:

- Fecha de registro
- Identificador del motor
- Temperatura
- Vibración
- Presión
- RPM
- Consumo energético
- Horas de operación
- Falla actual
- Indicador binario de falla en los próximos 15 días (variable objetivo)

## Enfoque de modelado

Se abordó el problema como una clasificación binaria (`falla_en_15_dias`: sí/no). El pipeline de Machine Learning incluye:

1. **Preprocesamiento**
   - Conversión de fechas
   - Ordenamiento por motor y fecha
   - Eliminación de registros incompletos

2. **Ingeniería de características**
   - Cálculo de medias móviles (7 días) para temperatura, vibración y consumo energético por motor
   - Generación de variables temporales: día, mes, día de la semana

3. **Entrenamiento del modelo**
   - Algoritmo utilizado: `XGBoostClassifier`
   - División en conjuntos de entrenamiento y prueba (70/30) con estratificación
   - Evaluación mediante F1-score, accuracy y matriz de confusión

4. **Rendimiento del modelo**
   - Accuracy: 73%
   - F1-score (clase positiva - falla): 0.66
   - Recall: 0.64
   - El modelo mostró buena capacidad de generalización con preferencia por variables acumuladas como vibración y temperatura.

## Aplicación

El modelo fue integrado en una aplicación web desarrollada con Streamlit, que permite:

- Cargar archivos `.csv` con datos nuevos
- Ejecutar predicciones motor por motor
- Visualizar resultados y descargar las predicciones

## Estructura del repositorio

```
├── app.py                   # Aplicación en Streamlit
├── models/
│   └── modelo_xgboost.pkl   # Modelo entrenado serializado
├── notebooks/               # Análisis exploratorio y desarrollo
├── src/                     # Scripts auxiliares (features, preprocesamiento)
├── data/                    # Datos de entrada (formato de ejemplo)
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Descripción técnica del proyecto
```

## Requisitos

- Python 3.10+
- pandas, scikit-learn, xgboost, streamlit, matplotlib

Instalación de dependencias:

```
pip install -r requirements.txt
```

## Ejecución

Desde la raíz del proyecto:

```
streamlit run app.py
```

## Autor

Matías Chamorro  
Certificación: Data Science II - Machine Learning aplicado a la ciencia de datos
