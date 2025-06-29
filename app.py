import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Cargar modelo entrenado
# ---------------------------
@st.cache(allow_output_mutation=True)
def cargar_modelo():
    with open("models/modelo_xgboost.pkl", "rb") as f:
        modelo = pickle.load(f)
    return modelo

modelo = cargar_modelo()

# ---------------------------
# Título y descripción
# ---------------------------
st.title("Predicción de Fallas en Motores")
st.markdown("""
Esta aplicación predice si un motor fallará en los próximos 15 días.
Subí un archivo `.csv` con datos nuevos de sensores y condiciones operativas.
""")

# ---------------------------
# Carga de archivo CSV
# ---------------------------
uploaded_file = st.file_uploader("Subí un archivo CSV", type="csv")

if uploaded_file is not None:
    datos_nuevos = pd.read_csv(uploaded_file)

    st.subheader("Datos cargados")
    st.dataframe(datos_nuevos.head())

    # Verificamos que no esté la columna 'fecha' y que esté el resto
    columnas_requeridas = [
        'motor_id', 'temperatura', 'vibracion', 'presion', 'rpm',
        'horas_operacion', 'consumo_energia',
        'temp_media_7d', 'vibra_media_7d', 'energia_media_7d'
    ]

    if all(col in datos_nuevos.columns for col in columnas_requeridas):
        # Generar predicciones
        X_nuevo = datos_nuevos[columnas_requeridas]
        predicciones = modelo.predict(X_nuevo)
        datos_nuevos['prediccion_falla_15_dias'] = predicciones

        st.subheader("Resultados de Predicción")
        st.write("0 = No fallará | 1 = Fallará en los próximos 15 días")
        st.dataframe(datos_nuevos[['motor_id', 'prediccion_falla_15_dias']])

        # Descargar resultados
        csv_resultado = datos_nuevos.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar resultados como CSV",
            data=csv_resultado,
            file_name="predicciones_falla.csv",
            mime="text/csv"
        )
    else:
        st.error("⚠️ El archivo no contiene todas las columnas requeridas.")
