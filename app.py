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
import os
from fpdf import FPDF
from datetime import datetime
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
tab1, tab2, tab3 = st.tabs(["Predicción Individual", "Predicción por Lotes", "Simulador"])

with tab1:
    # Formulario de entrada
    st.header("📊 Ingrese los parámetros de composición")

    # Sección para nombre y origen de la muestra
    with st.expander("Información de la muestra", expanded=True):
        col1, col2 = st.columns(2)
        sample_name = col1.text_input("Nombre de la muestra", "Muestra 1")
        origin = col2.text_input("Origen", "Agroresiduo")

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
        # Crear dataframe con los inputs (EXACTAMENTE como en tu versión original)
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
        st.success(f"**Valor FRAP predicho:** {prediction:.2f} mmol Fe2+/100g")

        # Clasificación
        if prediction > 50:
            classification = "Alto"
            interpretation = "Alto potencial funcional"
            recommendation = "Priorizar"
        elif prediction > 20:
            classification = "Medio"
            interpretation = "Potencial moderado"
            recommendation = "Considerar"
        else:
            classification = "Bajo"
            interpretation = "Bajo potencial"
            recommendation = "Descartar"

        st.info(f"""
        **Clasificación:** {classification}
        **Interpretación:** {interpretation}
        **Recomendación:** {recommendation}
        """)

        # Gráficos SHAP (Beeswarm y Waterfall)
        st.subheader("📊 Explicación del Modelo (SHAP)")

        # Calcular valores SHAP
        explainer = shap.Explainer(model, scaler.transform(np.zeros((1,8))))
        shap_values = explainer(scaled_data)

        # Gráfico Beeswarm
        st.markdown("**Importancia global de características**")
        fig1, ax1 = plt.subplots()
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig1)

        # Gráfico Waterfall
        st.markdown("**Contribución para esta predicción**")
        fig2, ax2 = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=8, show=False)
        st.pyplot(fig2)

        # Opciones de descarga (mejoradas)
        st.subheader("📥 Exportar resultados")

        # PDF con SHAP
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Información de la muestra
        pdf.cell(200, 10, txt="Reporte de Predicción FRAP", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Muestra: {sample_name}", ln=1)
        pdf.cell(200, 10, txt=f"Origen: {origin}", ln=1)
        pdf.cell(200, 10, txt=f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", ln=1)

        # Resultados
        pdf.cell(200, 10, txt=f"Valor FRAP predicho: {prediction:.2f} mmol Fe2+/100g", ln=1)
        pdf.cell(200, 10, txt=f"Clasificación: {classification}", ln=1)
        pdf.cell(200, 10, txt=f"Interpretación: {interpretation}", ln=1)
        pdf.cell(200, 10, txt=f"Recomendación: {recommendation}", ln=1)

        # Guardar gráficos temporalmente
        fig1.savefig("temp_beeswarm.png")
        fig2.savefig("temp_waterfall.png")

        # Añadir gráficos al PDF
        pdf.image("temp_beeswarm.png", x=10, y=60, w=180)
        pdf.image("temp_waterfall.png", x=10, y=140, w=180)

        # Generar PDF
        pdf_output = pdf.output(dest='S').encode('latin1')
        b64 = base64.b64encode(pdf_output).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="frap_report_{sample_name}.pdf">Descargar Reporte Completo (PDF)</a>'
        st.markdown(href, unsafe_allow_html=True)

        # CSV
        csv = input_data.copy()
        csv['FRAP_predicted'] = prediction
        csv['sample_name'] = sample_name
        csv['origin'] = origin
        csv_str = csv.to_csv(index=False)
        b64_csv = base64.b64encode(csv_str.encode()).decode()
        href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="frap_data_{sample_name}.csv">Descargar Datos (CSV)</a>'
        st.markdown(href_csv, unsafe_allow_html=True)

        # Limpiar archivos temporales
        #plt.close('all')
        #for f in ["temp_beeswarm.png", "temp_waterfall.png"]:
        #    if os.path.exists(f):
        #        os.remove(f)

with tab2:
    st.header("📁 Predicción por Lotes")
    st.markdown("Suba un archivo CSV con múltiples muestras para obtener predicciones masivas.")

    # CSV de ejemplo
    example_data = {
        'sample_name': ['Muestra1', 'Muestra2'],
        'origin': ['Origen1', 'Origen2'],
        'Moisture': [10.0, 12.0],
        'Protein': [15.0, 18.0],
        'Fat': [5.0, 4.0],
        'Total Carbohydrates': [60.0, 55.0],
        'Sugars': [10.0, 8.0],
        'Dietary Fiber': [5.0, 6.0],
        'Crude Fiber': [3.0, 4.0],
        'Ash': [2.0, 3.0]
    }
    example_df = pd.DataFrame(example_data)

    st.download_button(
        label="Descargar CSV de ejemplo",
        data=example_df.to_csv(index=False).encode('utf-8'),
        file_name="batch_example.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Seleccione archivo CSV", type="csv")

    if uploaded_file is not None:
        try:
            # Leer archivo
            batch_data = pd.read_csv(uploaded_file)

            # Verificar columnas (EXACTAMENTE como en tu versión original)
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
                        results['Clasificación'] = results['FRAP_predicted'].apply(
                            lambda x: "Alto" if x > 50 else "Medio" if x > 20 else "Bajo"
                        )

                    # Mostrar resultados
                    st.subheader("Resultados de Predicción")
                    st.dataframe(results)

                    # Gráfico resumen
                    st.subheader("Distribución de Valores FRAP Predichos")
                    fig, ax = plt.subplots()
                    sns.histplot(predictions, kde=True, ax=ax)
                    ax.set_xlabel("FRAP (mmol Fe2+/100g)")
                    st.pyplot(fig)

                    # SHAP para la primera muestra
                    st.subheader("Explicación para la primera muestra")
                    explainer = shap.Explainer(model, scaler.transform(np.zeros((1,8))))
                    shap_values = explainer(scaler.transform(X_batch.iloc[[0]]))

                    fig1, ax1 = plt.subplots()
                    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
                    st.pyplot(fig1)

                    # Descarga
                    st.subheader("📥 Exportar resultados")

                    # CSV
                    csv = results.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="batch_predictions.csv">Descargar Resultados (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    # PDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=10)

                    # Información general
                    pdf.cell(200, 10, txt="Reporte de Predicción FRAP por Lote", ln=1, align='C')
                    pdf.cell(200, 10, txt=f"Total muestras: {len(results)}", ln=1)
                    pdf.cell(200, 10, txt=f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", ln=1)

                    # Tabla de resumen
                    pdf.cell(200, 10, txt="Resumen estadístico:", ln=1)
                    pdf.cell(200, 10, txt=f"Mínimo: {results['FRAP_predicted'].min():.2f}", ln=1)
                    pdf.cell(200, 10, txt=f"Promedio: {results['FRAP_predicted'].mean():.2f}", ln=1)
                    pdf.cell(200, 10, txt=f"Máximo: {results['FRAP_predicted'].max():.2f}", ln=1)

                    # Gráfico
                    fig.savefig("temp_hist.png")
                    pdf.image("temp_hist.png", x=10, y=60, w=180)

                    # Generar PDF
                    pdf_output = pdf.output(dest='S').encode('latin1')
                    b64_pdf = base64.b64encode(pdf_output).decode()
                    href_pdf = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="batch_report.pdf">Descargar Reporte (PDF)</a>'
                    st.markdown(href_pdf, unsafe_allow_html=True)

                    # Limpiar temporal
                    plt.close('all')
                    if os.path.exists("temp_hist.png"):
                        os.remove("temp_hist.png")

            else:
                st.error(f"El archivo debe contener las columnas: {', '.join(required_cols)}")
                st.markdown("**Ejemplo de formato:**")
                st.code("""sample_name,origin,Moisture,Protein,Fat,Total Carbohydrates,Sugars,Dietary Fiber,Crude Fiber,Ash
Muestra1,Origen1,10.0,15.0,5.0,60.0,10.0,5.0,3.0,2.0""")

        except Exception as e:
            st.error(f"Error al procesar archivo: {str(e)}")

with tab3:
    st.header("🔍 Simulador What-If")
    st.markdown("Explore cómo cambios en los componentes afectan el valor FRAP predicho.")

    # Datos base
    base_data = {
        'Moisture': 10.0,
        'Protein': 15.0,
        'Fat': 5.0,
        'Total Carbohydrates': 60.0,
        'Sugars': 10.0,
        'Dietary Fiber': 5.0,
        'Crude Fiber': 3.0,
        'Ash': 2.0
    }

    # Selector de componente
    component = st.selectbox("Seleccione componente a modificar", list(base_data.keys()))

    # Control deslizante para modificar el valor
    change = st.slider(
        f"Cambio en {component} (unidades porcentuales)",
        -20.0, 20.0, 0.0, 0.5
    )

    if st.button("Calcular impacto"):
        with st.spinner('Calculando...'):
            # Crear versión modificada
            modified = base_data.copy()
            modified[component] += change
            modified[component] = max(0, modified[component])  # No permitir valores negativos

            # Predecir ambos casos
            input_original = pd.DataFrame([base_data])
            input_modified = pd.DataFrame([modified])

            scaled_original = scaler.transform(input_original)
            scaled_modified = scaler.transform(input_modified)

            pred_original = model.predict(scaled_original)[0][0]
            pred_modified = model.predict(scaled_modified)[0][0]

            # Mostrar resultados
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Valor original",
                    value=f"{pred_original:.2f} mmol Fe2+/100g",
                )
            with col2:
                st.metric(
                    label="Nuevo valor",
                    value=f"{pred_modified:.2f} mmol Fe2+/100g",
                    delta=f"{pred_modified - pred_original:+.2f}"
                )

            # Interpretación del cambio
            if pred_modified > pred_original:
                st.success("El cambio aumenta la actividad antioxidante")
            elif pred_modified < pred_original:
                st.warning("El cambio disminuye la actividad antioxidante")
            else:
                st.info("El cambio no afecta la actividad antioxidante")

# Sidebar
with st.sidebar:
    st.markdown("""
    ### 📚 Guía Rápida
    1. **Predicción Individual**: Ingrese valores manualmente
    2. **Predicción por Lotes**: Suba un archivo CSV
    3. **Simulador**: Explore cómo cambios afectan el FRAP
    """)

    st.markdown("### 🔍 Método FRAP")
    st.markdown("""
    El método FRAP (Ferric Reducing Antioxidant Power) mide capacidad antioxidante
    de una muestra para reducir los iones férricos (Fe3+) a iones ferrosos (Fe2+).
    """)

    st.markdown("### 📊 Clasificación FRAP")
    st.markdown("""
    - **Alto**: > 10 mmol Fe2+/100g
    - **Medio**: 2-10 mmol Fe2+/100g
    - **Bajo**: < 2 mmol Fe2+/100g
    """)
