import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import tensorflow as tf
from io import StringIO
import base64
from fpdf import FPDF
import time

# Configuración de la página
st.set_page_config(
    page_title="FRAP Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar modelo y scaler
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/frap_model.keras')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Título y descripción
st.title("🌱 AgroWaste-APP: Antioxidant Power Predictor")
st.markdown("""
**Aplicación para predecir actividad antioxidante (FRAP) a partir de la composición proximal de residuos agroindustriales**
""")

# Pestañas principales
tab1, tab2 = st.tabs(["Predicción Individual", "Predicción por Lotes"])

with tab1:
    # Formulario de entrada
    st.header("📊 Ingrese los parámetros de composición")
    col1, col2 = st.columns(2)

    with col1:
        moisture = st.number_input("Humedad (g/100g)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        protein = st.number_input("Proteína (g/100g)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        fat = st.number_input("Grasa (g/100g)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        carbs = st.number_input("Carbohidratos totales (g/100g)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

    with col2:
        sugars = st.number_input("Azúcares (g/100g)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        dietary_fiber = st.number_input("Fibra dietética (g/100g)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        crude_fiber = st.number_input("Fibra cruda (g/100g)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
        ash = st.number_input("Cenizas (g/100g)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

    # Botón de predicción
    if st.button("Predecir FRAP", type="primary"):
        # Crear dataframe con los inputs
        input_data = pd.DataFrame([[moisture, protein, fat, carbs, sugars, dietary_fiber, crude_fiber, ash]],
                                columns=['Moisture', 'Protein', 'Fat', 'Total Carbohydrates',
                                        'Sugars', 'Dietary Fiber', 'Crude Fiber', 'Ash'])

        # Escalar datos
        scaled_data = scaler.transform(input_data)

        # Realizar predicción
        with st.spinner('Calculando predicción...'):
            time.sleep(1)  # Simular procesamiento
            prediction = model.predict(scaled_data)[0][0]

        # Mostrar resultados
        st.success(f"**Valor FRAP predicho:** {prediction:.2f} mmol Fe²⁺/100g")

        # Gráfico de contribución con SHAP
        st.subheader("📌 Contribución de cada componente")

        # Calcular valores SHAP
        explainer = shap.Explainer(model, scaler.transform(np.zeros((1,8))))
        shap_values = explainer(scaled_data)

        # Visualización
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=8, show=False)
        plt.tight_layout()
        st.pyplot(fig)

        # Opciones de descarga
        st.subheader("📥 Exportar resultados")

        # Crear PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Reporte de Predicción FRAP", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Valor FRAP predicho: {prediction:.2f} mmol Fe²⁺/100g", ln=1)

        # Guardar gráfico temporalmente
        plt.savefig("temp_shap.png")
        pdf.image("temp_shap.png", x=10, y=40, w=180)

        # Generar enlace de descarga
        pdf_output = pdf.output(dest='S').encode('latin1')
        b64 = base64.b64encode(pdf_output).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="frap_prediction.pdf">Descargar Reporte (PDF)</a>'
        st.markdown(href, unsafe_allow_html=True)

        # CSV
        csv = input_data.copy()
        csv['FRAP_predicted'] = prediction
        csv_str = csv.to_csv(index=False)
        b64_csv = base64.b64encode(csv_str.encode()).decode()
        href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="frap_prediction.csv">Descargar Datos (CSV)</a>'
        st.markdown(href_csv, unsafe_allow_html=True)

with tab2:
    st.header("📁 Predicción por Lotes")
    st.markdown("Suba un archivo CSV con múltiples muestras para obtener predicciones masivas.")

    uploaded_file = st.file_uploader("Seleccione archivo CSV", type="csv")

    if uploaded_file is not None:
        try:
            # Leer archivo
            batch_data = pd.read_csv(uploaded_file)

            # Verificar columnas
            required_cols = ['Moisture', 'Protein', 'Fat', 'Total Carbohydrates',
                           'Sugars', 'Dietary Fiber', 'Crude Fiber', 'Ash']

            if all(col in batch_data.columns for col in required_cols):
                st.success("Archivo válido detectado")
                st.dataframe(batch_data.head())

                if st.button("Predecir Lote", type="primary"):
                    # Procesar datos
                    X_batch = batch_data[required_cols]
                    scaled_batch = scaler.transform(X_batch)

                    # Predecir
                    with st.spinner('Procesando muestras...'):
                        predictions = model.predict(scaled_batch).flatten()
                        results = batch_data.copy()
                        results['FRAP_predicted'] = predictions

                    # Mostrar resultados
                    st.subheader("Resultados de Predicción")
                    st.dataframe(results)

                    # Descarga
                    csv = results.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="batch_predictions.csv">Descargar Resultados (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    # Gráfico resumen
                    st.subheader("Distribución de Valores FRAP Predichos")
                    fig, ax = plt.subplots()
                    sns.histplot(predictions, kde=True, ax=ax)
                    ax.set_xlabel("FRAP (mmol Fe²⁺/100g)")
                    st.pyplot(fig)
            else:
                st.error(f"El archivo debe contener las columnas: {', '.join(required_cols)}")
                st.markdown("**Ejemplo de formato:**")
                st.code("""Moisture,Protein,Fat,Total Carbohydrates,Sugars,Dietary Fiber,Crude Fiber,Ash
10.0,15.0,5.0,60.0,10.0,5.0,3.0,2.0""")

        except Exception as e:
            st.error(f"Error al procesar archivo: {str(e)}")

# Sidebar con información adicional
with st.sidebar:
    #st.image("assets/logo.png", width=200)
    st.markdown("""
    ### 📚 Guía Rápida
    1. **Predicción Individual**: Ingrese valores manualmente
    2. **Predicción por Lotes**: Suba un archivo CSV
    3. Los resultados pueden exportarse como PDF o CSV
    """)

    st.markdown("### 🔍 Método FRAP")
    st.markdown("""
    El método FRAP (Ferric Reducing Ability of Plasma) mide la capacidad antioxidante
    mediante la reducción de iones férricos a ferrosos.
    """)

    st.markdown("### 📊 Datos de Ejemplo")
    with open("sample_data/example_batch.csv", "rb") as file:
        btn = st.download_button(
            label="Descargar CSV de ejemplo",
            data=file,
            file_name="example_batch.csv",
            mime="text/csv"
        )
