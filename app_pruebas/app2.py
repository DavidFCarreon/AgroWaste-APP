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
from datetime import datetime
from jinja2 import Template
from fpdf import FPDF
import time

# Page Configuration
st.set_page_config(
    page_title="FRAP Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/frap_model.keras')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Helper functions
def get_clean_feature_names():
    return ['Moisture', 'Protein', 'Fat', 'Ash', 'Crude Fiber',
            'Total Carbohydrates', 'Dietary Fiber', 'Sugars']

def predict_frap(input_data):
    """Wrapper function for model prediction"""
    if isinstance(input_data, pd.DataFrame):
        scaled_data = scaler.transform(input_data)
    else:  # assuming it's a dictionary
        input_df = pd.DataFrame([input_data])
        scaled_data = scaler.transform(input_df)
    return model.predict(scaled_data)[0][0]

def generate_shap_plots(input_data, sample_name):
    """Generate SHAP plots and return image paths"""
    if isinstance(input_data, pd.DataFrame):
        scaled_data = scaler.transform(input_data)
    else:
        input_df = pd.DataFrame([input_data])
        scaled_data = scaler.transform(input_df)

    explainer = shap.Explainer(model, scaler.transform(np.zeros((1,8))))
    shap_values = explainer(scaled_data)
    shap_values[0].feature_names = get_clean_feature_names()

    # Beeswarm plot
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    beeswarm_path = f"shap_beeswarm_{sample_name}.png"
    plt.savefig(beeswarm_path, bbox_inches='tight', dpi=150)
    plt.close()

    # Waterfall plot
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    waterfall_path = f"shap_waterfall_{sample_name}.png"
    plt.savefig(waterfall_path, bbox_inches='tight', dpi=150)
    plt.close()

    return beeswarm_path, waterfall_path

def generate_report_with_shap(data, frap_value, beeswarm_img, waterfall_img):
    """Generate PDF report with SHAP visualizations"""
    classification = "Alto" if frap_value > 50 else "Medio" if frap_value > 20 else "Bajo"
    interpretation = {
        "Alto": "Alto potencial funcional",
        "Medio": "Potencial moderado",
        "Bajo": "Bajo potencial"
    }[classification]
    recommendation = {
        "Alto": "Priorizar",
        "Medio": "Considerar",
        "Bajo": "Descartar"
    }[classification]

    try:
        # Convert images to base64 for HTML embedding
        def img_to_base64(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        beeswarm_b64 = img_to_base64(beeswarm_img)
        waterfall_b64 = img_to_base64(waterfall_img)

        # HTML template with embedded images
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Informe FRAP - {data['sample_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 30px; }}
                h1, h2 {{ color: #2C3E50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #F2F2F2; }}
                .highlight {{ background-color: #E8F5E8; padding: 10px; border-left: 4px solid green; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .img-caption {{ font-size: 0.9em; color: #555; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Informe de Evaluación de Potencial Antioxidante</h1>
            <p><strong>Muestra:</strong> {data['sample_name']}</p>
            <p><strong>Origen:</strong> {data.get('origin', 'N/A')}</p>
            <p><strong>Fecha:</strong> {datetime.now().strftime("%d/%m/%Y")}</p>

            <h2>Predicción de Actividad Antioxidante (FRAP)</h2>
            <div class="highlight">
                <p><strong>FRAP predicho:</strong> {frap_value:.2f} mmol Fe2+/100g</p>
                <p><strong>Clasificación:</strong> {classification}</p>
                <p><strong>Interpretación:</strong> {interpretation}</p>
                <p><strong>Recomendación:</strong> {recommendation}</p>
            </div>

            <h2>Composición Proximal</h2>
            <table>
                <tr><th>Componente</th><th>Valor (%)</th></tr>
                <tr><td>Humedad</td><td>{data['moisture']}</td></tr>
                <tr><td>Proteína</td><td>{data['protein']}</td></tr>
                <tr><td>Grasa</td><td>{data['fat']}</td></tr>
                <tr><td>cenizas</td><td>{data['ash']}</td></tr>
                <tr><td>fibra cruda</td><td>{data['crude_fiber']}</td></tr>
                <tr><td>carbohidratos totales</td><td>{data['total_carbohydrates']}</td></tr>
                <tr><td>fibra dietética</td><td>{data['dietary_fiber']}</td></tr>
                <tr><td>azúcares</td><td>{data['sugars']}</td></tr>
            </table>

            <h2>Explicabilidad del modelo (SHAP)</h2>
            <p>Los siguientes gráficos muestran cómo cada componente de la composición proximal contribuyó a la predicción del FRAP.</p>

            <h3>SHAP Beeswarm: Importancia global de features</h3>
            <img src="data:image/png;base64,{beeswarm_b64}" alt="SHAP Beeswarm">
            <p class="img-caption">Cada punto representa una predicción. Posición horizontal indica impacto en FRAP.</p>

            <h3>SHAP Waterfall: Desglose de esta predicción</h3>
            <img src="data:image/png;base64,{waterfall_b64}" alt="SHAP Waterfall">
            <p class="img-caption">Desglose paso a paso desde el valor base hasta la predicción final.</p>
        </body>
        </html>
        """

        # Save temporary HTML
        with open("informe_temp.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        # Convert to PDF
        from weasyprint import HTML
        HTML("informe_temp.html").write_pdf("informe_con_shap.pdf")

        # Read PDF
        with open("informe_con_shap.pdf", "rb") as f:
            pdf_data = f.read()

        # Clean temporary files
        for file in ["informe_temp.html", beeswarm_img, waterfall_img, "informe_con_shap.pdf"]:
            if os.path.exists(file):
                os.remove(file)

        return pdf_data

    except ImportError:
        st.warning("Instala weasyprint: pip install weasyprint")
        return None
    except Exception as e:
        st.error(f"Error al generar PDF: {e}")
        return None

# Main App
st.title("🌱 AgroWaste-APP: Antioxidant Power Predictor")
st.markdown("""
**Aplicación para predecir actividad antioxidante (FRAP) a partir de la composición proximal de residuos agroindustriales**
""")

# Main tabs
tab1, tab2, tab3 = st.tabs(["Predicción Individual", "Predicción por Lotes", "Simulador"])

with tab1:
    st.header("📊 Ingrese los parámetros de composición")

    with st.expander("Datos de la muestra"):
        col1, col2 = st.columns(2)
        sample_name = col1.text_input("Nombre de la muestra", "Muestra 1")
        origin = col2.text_input("Origen", "Agroresiduo")

    col1, col2 = st.columns(2)
    with col1:
        moisture = st.number_input("Humedad (g/100g)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        protein = st.number_input("Proteína (g/100g)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        fat = st.number_input("Grasa (g/100g)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        ash = st.number_input("Cenizas (g/100g)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

    with col2:
        crude_fiber = st.number_input("Fibra cruda (g/100g)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
        total_carbohydrates = st.number_input("Carbohidratos totales (g/100g)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
        dietary_fiber = st.number_input("Fibra dietética (g/100g)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        sugars = st.number_input("Azúcares (g/100g)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    if st.button("Predecir FRAP", type="primary", key="predict_single"):
        input_data = {
            'moisture': moisture,
            'protein': protein,
            'fat': fat,
            'ash': ash,
            'crude_fiber': crude_fiber,
            'total_carbohydrates': total_carbohydrates,
            'dietary_fiber': dietary_fiber,
            'sugars': sugars,
            'sample_name': sample_name,
            'origin': origin
        }

        with st.spinner('Calculando predicción...'):
            time.sleep(1)
            prediction = predict_frap(input_data)

            # Generate SHAP plots
            beeswarm_path, waterfall_path = generate_shap_plots(input_data, sample_name)

            # Display results
            st.success(f"**Valor FRAP predicho:** {prediction:.2f} mmol Fe²⁺/100g")

            # Classification
            classification = "Alto" if prediction > 50 else "Medio" if prediction > 20 else "Bajo"
            st.info(f"**Clasificación:** {classification}")

            # SHAP visualizations
            st.subheader("Explicación del Modelo (SHAP)")

            col1, col2 = st.columns(2)
            with col1:
                st.image(beeswarm_path, caption="Importancia global de features (Beeswarm)")
            with col2:
                st.image(waterfall_path, caption="Desglose de esta predicción (Waterfall)")

            # Generate and offer download of report
            pdf_data = generate_report_with_shap(
                data=input_data,
                frap_value=prediction,
                beeswarm_img=beeswarm_path,
                waterfall_img=waterfall_path
            )

            if pdf_data:
                st.download_button(
                    "Descargar informe completo (PDF)",
                    pdf_data,
                    file_name=f"informe_frap_{sample_name}.pdf",
                    mime="application/pdf"
                )

with tab2:
    st.header("📁 Predicción por Lotes")
    st.markdown("Suba un archivo CSV con múltiples muestras para obtener predicciones masivas.")

    # Example CSV download
    try:
        with open("sample_data/example_batch.csv", "rb") as file:
            st.download_button(
                label="Descargar CSV de ejemplo",
                data=file,
                file_name="example_batch.csv",
                mime="text/csv"
            )
    except:
        pass

    uploaded_file = st.file_uploader("Seleccione archivo CSV", type="csv")

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)

            # Check for required columns
            required_cols = ['moisture', 'protein', 'fat', 'ash',
                           'crude_fiber', 'total_carbohydrates',
                           'dietary_fiber', 'sugars']

            if all(col in batch_data.columns for col in required_cols):
                st.success("Archivo válido detectado")
                st.dataframe(batch_data.head())

                if st.button("Predecir Lote", type="primary", key="predict_batch"):
                    with st.spinner('Procesando muestras...'):
                        # Add predictions
                        batch_data['FRAP_predicho'] = batch_data.apply(predict_frap, axis=1)
                        batch_data['Clasificación'] = batch_data['FRAP_predicho'].apply(
                            lambda x: "Alto" if x > 50 else "Medio" if x > 20 else "Bajo"
                        )

                        # Generate SHAP explanations for the first sample
                        sample_row = batch_data.iloc[0].to_dict()
                        beeswarm_path, waterfall_path = generate_shap_plots(sample_row, "batch_sample")

                    # Show results
                    st.subheader("Resultados de Predicción")
                    st.dataframe(batch_data.sort_values("FRAP_predicho", ascending=False))

                    # Summary visualization
                    st.subheader("Distribución de Valores FRAP Predichos")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(data=batch_data, x=batch_data.index, y='FRAP_predicho',
                                hue='Clasificación', dodge=False, ax=ax)
                    plt.xticks(rotation=45)
                    ax.set_title("FRAP predicho por muestra")
                    st.pyplot(fig)

                    # SHAP visualizations
                    st.subheader("Explicación del Modelo (Primera muestra)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(beeswarm_path, caption="Importancia global de features (Beeswarm)")
                    with col2:
                        st.image(waterfall_path, caption="Desglose de predicción (Waterfall)")

                    # Download options
                    st.subheader("📥 Exportar resultados")

                    # CSV download
                    csv = batch_data.to_csv(index=False)
                    b64_csv = base64.b64encode(csv.encode()).decode()
                    href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="batch_predictions.csv">Descargar Resultados (CSV)</a>'
                    st.markdown(href_csv, unsafe_allow_html=True)

                    # PDF report for batch
                    if st.button("Generar informe PDF por lote"):
                        df_html = batch_data.to_html(index=False)
                        html = f"""<!DOCTYPE html><html><body>
                        <h1>Informe por Lote - FRAP Predicho</h1>
                        <p>Total muestras: {len(batch_data)}</p>
                        {df_html}</body></html>"""

                        with open("lote_temp.html", "w") as f:
                            f.write(html)

                        from weasyprint import HTML
                        HTML("lote_temp.html").write_pdf("informe_lote.pdf")

                        with open("informe_lote.pdf", "rb") as f:
                            pdf_data = f.read()

                        os.remove("lote_temp.html")

                        st.download_button(
                            "Descargar informe por lote (PDF)",
                            pdf_data,
                            file_name="informe_lote.pdf",
                            mime="application/pdf"
                        )

            else:
                st.error(f"El archivo debe contener las columnas: {', '.join(required_cols)}")
                st.markdown("**Ejemplo de formato:**")
                st.code("""sample_name,origin,moisture,protein,fat,ash,crude_fiber,total_carbohydrates,dietary_fiber,sugars
Muestra1,Agroresiduo,10.0,15.0,5.0,2.0,3.0,60.0,5.0,10.0""")

        except Exception as e:
            st.error(f"Error al procesar archivo: {str(e)}")

with tab3:
    st.header("🔍 Simulador What-If")
    st.markdown("Explore cómo cambios en los componentes afectan el valor FRAP predicho.")

    if 'batch_data' in locals() and len(batch_data) > 0:
        base_sample = st.selectbox("Seleccione muestra base", batch_data['sample_name'])
        base_row = batch_data[batch_data['sample_name'] == base_sample].iloc[0]
    else:
        # Default values if no batch data is loaded
        base_row = {
            'moisture': 10.0,
            'protein': 15.0,
            'fat': 5.0,
            'ash': 2.0,
            'crude_fiber': 3.0,
            'total_carbohydrates': 60.0,
            'dietary_fiber': 5.0,
            'sugars': 10.0
        }

    component = st.selectbox("Componente a modificar", [
        'moisture', 'protein', 'fat', 'ash',
        'crude_fiber', 'total_carbohydrates',
        'dietary_fiber', 'sugars'
    ])

    change = st.slider(
        f"Cambio en {component} (unidades porcentuales)",
        -20.0, 20.0, 0.0, 0.5
    )

    modified = base_row.copy()
    modified[component] += change
    modified[component] = max(0, modified[component])  # Ensure no negative values

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Valor original", f"{base_row[component]:.1f}")
    with col2:
        st.metric("Nuevo valor", f"{modified[component]:.1f}", delta=f"{change:+.1f}")

    if st.button("Calcular impacto en FRAP", key="whatif_calc"):
        with st.spinner('Calculando...'):
            frap_orig = predict_frap(base_row)
            frap_mod = predict_frap(modified)

            st.metric("FRAP original", f"{frap_orig:.2f} mmol Fe²⁺/100g")
            st.metric("FRAP modificado", f"{frap_mod:.2f} mmol Fe²⁺/100g",
                     delta=f"{frap_mod - frap_orig:+.2f}")

            if frap_mod > frap_orig:
                st.success("Aumento en actividad antioxidante")
            elif frap_mod < frap_orig:
                st.warning("Disminución en actividad antioxidante")
            else:
                st.info("Sin cambio en actividad antioxidante")

# Sidebar
with st.sidebar:
    st.markdown("""
    ### 📚 Guía Rápida
    1. **Predicción Individual**: Ingrese valores manualmente
    2. **Predicción por Lotes**: Suba un archivo CSV
    3. **Simulador**: Explore cómo cambios afectan el FRAP
    4. Exporte resultados como PDF o CSV
    """)

    st.markdown("### 🔍 Método FRAP")
    st.markdown("""
    El método FRAP (Ferric Reducing Ability of Plasma) mide la capacidad antioxidante
    mediante la reducción de iones férricos a ferrosos.
    """)

    st.markdown("### 📊 Clasificación FRAP")
    st.markdown("""
    - **Alto**: > 50 mmol Fe²⁺/100g
    - **Medio**: 20-50 mmol Fe²⁺/100g
    - **Bajo**: < 20 mmol Fe²⁺/100g
    """)
