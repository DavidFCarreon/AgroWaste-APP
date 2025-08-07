# -*- coding: utf-8 -*-
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import shap
import os
import streamlit as st
import requests
import matplotlib.pyplot as plt


# Rutas
MODEL_PATH = "AgroWaste_App/models/ml_model.pkl"
DATA_PATH = "AgroWaste_App/dataset/data_preprocessed_FRAP_final.csv"
BACKGROUND_PATH = "AgroWaste_App/models/background_df.pkl"



# Validar archivos
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH}")
if not os.path.exists(BACKGROUND_PATH):
    raise FileNotFoundError(f"No se encontró el dataset en {BACKGROUND_PATH}")

def get_clean_feature_names():
    """
    Devuelve los nombres limpios de las características para visualización.
    """
    return [
        "Humedad",
        "Proteína",
        "Grasa",
        "Ceniza",
        "Fibra Cruda",
        "Carb. Totales",
        "Fibra Dietética",
        "Azúcares"
    ]

# Cargar modelo
model = joblib.load(MODEL_PATH)
preprocessor = model.named_steps['preprocessor']

# Cargar datos para background
df_background = pd.read_csv(DATA_PATH)
X_background = df_background.iloc[:, :-1]  # Sin la columna FRAP (target)

# Transformar con el preprocessor del modelo
X_background_transformed = preprocessor.transform(X_background)

# Muestra 100 filas representativas
X_sampled_transformed = shap.sample(X_background_transformed, 100)  # Es un numpy array

# === 🔴 Corrección clave: convertir a DataFrame con nombres de columnas ===
feature_names = X_background.columns.tolist()  # Nombres originales de las columnas
clean_feature_names = get_clean_feature_names()  # Nombres limpios: Humedad, Proteína...

X_sampled_df = pd.DataFrame(X_sampled_transformed, columns=feature_names)

# Crear explainer: ahora usamos el DataFrame original (X_background) para el background
explainer = shap.Explainer(
    model.predict,
    X_sampled_df,
    feature_names=feature_names
)


def safe_api_request(url, params, max_retries=2, timeout=10):
    import time
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)  # Pequeño delay entre reintentos
    return None


def get_shap_explainer():
    return explainer


def get_background_data():
    """Devuelve el background usado para el explainer (para beeswarm global)"""
    return X_sampled_df

# --- Función para procesar múltiples muestras y generar SHAP ---
def process_batch_shap(df, api_url):
    """
    Procesa un lote de muestras y genera visualizaciones SHAP
    Args:
        df: DataFrame con las muestras
        api_url: URL de la API para predicciones
    Returns:
        Tuple: (shap_values_global, waterfall_images)
    """
    shap_values_list = []
    base_values_list = []
    feature_data_list = []
    waterfall_images = []

    with st.spinner("Calculando explicaciones SHAP..."):
        progress_bar = st.progress(0)
        total_samples = len(df)

        for idx, row in df.iterrows():
            # Sanitizar el nombre de la muestra para el nombre de archivo
            safe_name = f"sample_{idx}"  # Usamos el índice como identificador único
            # Obtener predicción y valores SHAP de la API
            params = {
                'Moisture': row['moisture'],
                'Protein': row['protein'],
                'Fat': row['fat'],
                'Ash': row['ash'],
                'Crude_Fiber': row['crude_fiber'],
                'Total_Carbohydrates': row['total_carbohydrates'],
                'Dietary_Fiber': row['dietary_fiber'],
                'Sugars': row['sugars']
            }
            try:
                response = requests.get(api_url, params=params)
                if response.status_code == 200:
                    result = response.json()

                    # Acumular datos para el beeswarm global
                    shap_values_list.append(result['shap_values'][0])  # Tomar el primer elemento del array
                    base_values_list.append(result['shap_base_values'][0])
                    feature_data_list.append(result['shap_data'][0])

                    # Crear waterfall para esta muestra
                    explanation = shap.Explanation(
                        values=np.array([result['shap_values'][0]]),
                        base_values=np.array([result['shap_base_values'][0]]),
                        data=np.array([result['shap_data'][0]]),
                        feature_names=get_clean_feature_names()
                    )

                    # Guardar imagen del waterfall
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    shap.plots.waterfall(explanation[0], show=False)
                    # Ajustar márgenes para optimizar espacio
                    plt.tight_layout()
                    img_path = f"shap_waterfall_{safe_name}.png"
                    fig.savefig(img_path, bbox_inches='tight', dpi=130, facecolor='white')
                    plt.close(fig)
                    waterfall_images.append((img_path, row['sample_name']))  # Guardamos ambos: path y nombre original

                progress_bar.progress((idx + 1) / total_samples)

            except Exception as e:
                st.warning(f"Error procesando muestra {row['sample_name']}: {str(e)}")
                continue

    # Crear objeto Explanation global para el beeswarm
    shap_values_global = shap.Explanation(
        values=np.array(shap_values_list),
        base_values=np.array(base_values_list),
        data=np.array(feature_data_list),
        feature_names=get_clean_feature_names()
    )
    return shap_values_global, waterfall_images

# --- Función generate_report_with_shap (optimizada) ---
def generate_report_with_shap(data, frap_value, beeswarm_img, waterfall_img, recommendations=None):
    import os
    import base64
    import pdfkit
    import tempfile
    from datetime import datetime

    classification = "Alto" if frap_value > 40 else "Medio" if frap_value > 15 else "Bajo"
    interpretation = {"Alto": "Alto potencial funcional", "Medio": "Potencial moderado", "Bajo": "Bajo potencial"}[classification]
    recommendation = {"Alto": "Priorizar", "Medio": "Considerar", "Bajo": "Descartar"}[classification]

    try:
        def img_to_base64(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        beeswarm_b64 = img_to_base64(beeswarm_img)
        waterfall_b64 = img_to_base64(waterfall_img)

        recommendations_section = ""
        if recommendations:
            recommendations_section = f"""
            <div style="page-break-before: always;">
                <h2> Recomendaciones de Investigación y Desarrollo</h2>
                <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #4CAF50; border-radius: 5px; margin-bottom: 20px;">
                    <h3 style="color: #2c3e50; margin-top: 0;">Oportunidades de valorización para {data['sample_name']}</h3>
                    <div style="font-size: 0.95em; line-height: 1.6; text-align: justify;">
                        {recommendations}
                    </div>
                </div>
            </div>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Informe FRAP - {data['sample_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 30px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #e8f5e8; padding: 10px; border-left: 4px solid green; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .img-caption {{ font-size: 0.9em; color: #555; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Informe de Evaluación de Potencial Antioxidante</h1>
            <p><strong>Muestra:</strong> {data['sample_name']}</p>
            <p><strong>Origen:</strong> {data.get('origin', 'N/A')}</p>
            <p><strong>Fecha:</strong> {datetime.now().strftime('%d/%m/%Y')}</p>
            <div class="highlight">
                <p><strong>FRAP predicho:</strong> {frap_value:.2f} mmol Fe2+/100g</p>
                <p><strong>Clasificación:</strong> {classification}</p>
                <p><strong>Interpretación:</strong> {interpretation}</p>
                <p><strong>Recomendación general:</strong> {recommendation}</p>
            </div>
            <h2>Composición Proximal</h2>
            <table>
                <tr><th>Componente</th><th>Valor (%)</th></tr>
                <tr><td>Humedad</td><td>{data['moisture']}</td></tr>
                <tr><td>Proteína</td><td>{data['protein']}</td></tr>
                <tr><td>Grasa</td><td>{data['fat']}</td></tr>
                <tr><td>Cenizas</td><td>{data['ash']}</td></tr>
                <tr><td>Fibra cruda</td><td>{data['crude_fiber']}</td></tr>
                <tr><td>Carbohidratos totales</td><td>{data['total_carbohydrates']}</td></tr>
                <tr><td>Fibra dietética</td><td>{data['dietary_fiber']}</td></tr>
                <tr><td>Azúcares</td><td>{data['sugars']}</td></tr>
            </table>
            <h2>Explicabilidad del modelo (SHAP)</h2>
            <h3>🐝 SHAP Beeswarm</h3>
            <img src="data:image/png;base64,{beeswarm_b64}" alt="SHAP Beeswarm">
            <p class="img-caption">Importancia de features en múltiples predicciones.</p>
            <h3>📊 SHAP Waterfall</h3>
            <img src="data:image/png;base64,{waterfall_b64}" alt="SHAP Waterfall">
            <p class="img-caption">Contribución paso a paso hacia la predicción final.</p>
            {recommendations_section}
        </body>
        </html>
        """

        # Guardar HTML temporal
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(html_content)
            html_path = f.name

        # Ruta de salida
        pdf_path = "informe_con_shap.pdf"

        # Generar PDF
        try:
            pdfkit.from_file(html_path, pdf_path)
        except Exception as e:
            print(f"Error en pdfkit: {e}")
            return None

        # Leer PDF
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        # Limpiar archivos
        for file in [html_path, pdf_path, "shap_beeswarm.png", "shap_waterfall.png"]:
            if os.path.exists(file):
                os.remove(file)

        return pdf_data

    except Exception as e:
        print(f"❌ Error al generar informe: {e}")
        return None

# --- Función generate_batch_report_with_shap (modificada con orientación horizontal) ---
def generate_batch_report_with_shap(df, waterfall_images):
    import os
    import base64
    import pdfkit
    import tempfile
    from datetime import datetime

    required_cols = ['moisture','protein','fat','ash','crude_fiber','total_carbohydrates','dietary_fiber','sugars']
    COLUMN_TRANSLATIONS = {
        'sample_name': 'Residuo agroindustrial',
        'origin': 'Origen',
        'moisture': 'Humedad',
        'protein': 'Proteína',
        'fat': 'Grasa',
        'ash': 'Ceniza',
        'crude_fiber': 'Fibra Cruda',
        'total_carbohydrates': 'Carb. Totales',
        'dietary_fiber': 'Fibra Diet.',
        'sugars': 'Azúcares',
        'FRAP_predicho': 'FRAP Predicho',
        'Clasificación': 'Clasificación'
    }

    try:
        # Preparar datos
        input_batch = df[required_cols]
        explainer = get_shap_explainer()
        shap_values_batch = explainer(input_batch)
        shap_values_batch.feature_names = get_clean_feature_names()

        # --- Beeswarm para lote ---
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        shap.plots.beeswarm(shap_values_batch, show=False)
        plt.savefig("shap_beeswarm_lote.png", bbox_inches='tight', dpi=150, facecolor='white')
        plt.close(fig1)

        # Convertir imágenes a base64
        def img_to_base64(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        beeswarm_b64 = img_to_base64("shap_beeswarm_lote.png")

        # Calcular estadísticas
        avg_frap = df['FRAP_predicho'].mean()
        high_count = len(df[df['Clasificación'] == 'Alto'])
        med_count = len(df[df['Clasificación'] == 'Medio'])
        low_count = len(df[df['Clasificación'] == 'Bajo'])

        # Traducir columnas para mostrar
        display_df = df.copy()
        display_df = display_df.rename(columns=COLUMN_TRANSLATIONS)
        df_html = display_df.to_html(index=False, escape=False)

        # Sección de recomendaciones
        recomendaciones_html = """
        <div style="page-break-before: always;">
            <h2 style="text-align: center;">Recomendaciones para I+D</h2>
            <div style="text-align: justify; margin: 0 5%;">
                <p>Cuando se evalúa un conjunto diverso de residuos agroindustriales con fines de valorización...</p>
        """

        if high_count > 0:
            recomendaciones_html += f"""
                <h3 style="text-align: center;">Residuos con Alta Capacidad Antioxidante: {high_count}</h3>
                <p><strong style="text-align: center; display: block;">(FRAP > 10 mmol Fe²⁺/100 g)</strong></p>
                <p><strong>Estrategia sugerida:</strong> Priorizar la recuperación de compuestos bioactivos antioxidantes...</p>
                <ul style="text-align: justify;">
                    <li>Extracción de compuestos fenólicos mediante tecnologías verdes</li>
                    <li>Desarrollo de ingredientes funcionales para alimentos y nutracéuticos</li>
                    <li>Aplicaciones en cosmética natural como antioxidantes</li>
                    <li>Microencapsulación para mejorar estabilidad y biodisponibilidad</li>
                    <li>Evaluación sinérgica con otros antioxidantes naturales</li>
                </ul>
            """

        if med_count > 0:
            recomendaciones_html += f"""
                <h3 style="text-align: center;">Residuos con Capacidad Antioxidante Media: {med_count}</h3>
                <p><strong style="text-align: center; display: block;">(FRAP entre 2 y 10 mmol Fe²⁺/100 g)</strong></p>
                <p><strong>Estrategia sugerida:</strong> Considerar una valorización dual o integrada...</p>
                <ul style="text-align: justify;">
                    <li>Desarrollo de extractos con funcionalidad moderada...</li>
                    <li>Incorporación como ingrediente funcional con aporte antioxidante complementario...</li>
                    <li>Evaluación como fuente de fibra dietética u otros metabolitos secundarios...</li>
                    <li>Uso como sustrato en procesos biotecnológicos...</li>
                    <li>Aplicación en formulación de productos combinados...</li>
                </ul>
            """

        if low_count > 0:
            recomendaciones_html += f"""
                <h3 style="text-align: center;">Residuos con Baja Capacidad Antioxidante: {low_count}</h3>
                <p><strong style="text-align: center; display: block;">(FRAP < 2 mmol Fe²⁺/100 g)</strong></p>
                <p><strong>Estrategia sugerida:</strong> Desviar el enfoque hacia otras fracciones...</p>
                <ul style="text-align: justify;">
                    <li>Aprovechamiento como fuente de fibra estructural...</li>
                    <li>Producción de biocombustibles o bioenergía...</li>
                    <li>Uso en alimentación animal, compostaje o formulación de enmiendas orgánicas...</li>
                    <li>Aplicaciones en fermentación de estado sólido o líquido...</li>
                    <li>Considerar su inclusión como componente de mezclas multirresiduo...</li>
                </ul>
            """

        recomendaciones_html += """
            </div>
        </div>
        """

        # Generar HTML de waterfalls (2 por página)
        waterfalls_html = ""
        for i in range(0, len(waterfall_images), 2):
            img_path_1 = waterfall_images[i][0]
            sample_name_1 = waterfall_images[i][1]
            second_image_html = ""
            if i + 1 < len(waterfall_images):
                img_path_2 = waterfall_images[i+1][0]
                sample_name_2 = waterfall_images[i+1][1]
                second_image_html = f"""
                <div style="width: 48%; margin-left: 2%;">
                    <h3>📊 SHAP Waterfall: {sample_name_2}</h3>
                    <div style="text-align: center;">
                        <img src="data:image/png;base64,{img_to_base64(img_path_2)}" alt="SHAP Waterfall" style="max-width: 95%; height: auto;">
                    </div>
                    <p class="img-caption">Desglose de la predicción para la muestra {sample_name_2}.</p>
                </div>
                """
            waterfalls_html += f"""
            <div style="page-break-before: always;">
                <div style="display: flex; justify-content: space-between;">
                    <div style="width: 48%; margin-right: 2%;">
                        <h3>📊 SHAP Waterfall: {sample_name_1}</h3>
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{img_to_base64(img_path_1)}" alt="SHAP Waterfall" style="max-width: 95%; height: auto;">
                        </div>
                        <p class="img-caption">Desglose de la predicción para la muestra {sample_name_1}.</p>
                    </div>
                    {second_image_html}
                </div>
            </div>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Informe por Lote - FRAP Predicho</title>
            <style>
                @page {{
                    size: landscape;
                    margin: 1cm;
                }}
                body {{ font-family: Arial, sans-serif; margin: 30px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .img-caption {{ font-size: 0.9em; color: #555; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Informe de Análisis por Lote - FRAP Predicho</h1>
            <p><strong>Fecha:</strong> {datetime.now().strftime('%d/%m/%Y')}</p>
            <p><strong>Total de muestras:</strong> {len(df)}</p>
            <p><strong>FRAP promedio:</strong> {avg_frap:.2f} mmol Fe²⁺/100g</p>
            <h2>Resultados</h2>
            {df_html}
            {recomendaciones_html}
            <h3>🐝 SHAP Beeswarm: Importancia global de features</h3>
            <img src="data:image/png;base64,{beeswarm_b64}" alt="SHAP Beeswarm">
            <p class="img-caption">Importancia de cada componente en el conjunto de predicciones.</p>
            {waterfalls_html}
        </body>
        </html>
        """

        # Guardar HTML temporal
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(html_content)
            html_path = f.name

        # Generar PDF
        pdf_path = "informe_lote_con_shap.pdf"
        pdfkit.from_file(html_path, pdf_path)

        # Leer PDF
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        # Limpiar archivos
        files_to_remove = [html_path, pdf_path, "shap_beeswarm_lote.png"]
        files_to_remove.extend([img_path for img_path, _ in waterfall_images])
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)

        return pdf_data

    except Exception as e:
        print(f"❌ Error al generar informe por lote: {e}")
        return None
