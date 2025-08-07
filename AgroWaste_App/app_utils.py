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
from xhtml2pdf import pisa
from io import BytesIO


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

def generate_report_with_shap(data, frap_value, beeswarm_img, waterfall_img, recommendations=None):
    import base64
    from io import BytesIO
    from xhtml2pdf import pisa

    classification = "Alto" if frap_value > 40 else "Medio" if frap_value > 15 else "Bajo"
    interpretation = {"Alto": "Alto potencial funcional",
                     "Medio": "Potencial moderado",
                     "Bajo": "Bajo potencial"}[classification]

    try:
        # Convertir imágenes a base64
        def img_to_base64(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        beeswarm_b64 = img_to_base64(beeswarm_img)
        waterfall_b64 = img_to_base64(waterfall_img)

        # Plantilla HTML con estilos optimizados
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Informe FRAP - {data['sample_name']}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                    margin-top: 25px;
                }}
                .header {{
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                .highlight {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-left: 4px solid #4CAF50;
                    border-radius: 4px;
                    margin: 20px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    margin: 15px 0;
                    border: 1px solid #ddd;
                }}
                .img-caption {{
                    font-style: italic;
                    text-align: center;
                    color: #666;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Informe de Evaluación de Potencial Antioxidante</h1>
                <p><strong>Muestra:</strong> {data['sample_name']}</p>
                <p><strong>Fecha:</strong> {datetime.now().strftime('%d/%m/%Y')}</p>
            </div>

            <div class="highlight">
                <p><strong>FRAP predicho:</strong> {frap_value:.2f} mmol Fe²⁺/100g</p>
                <p><strong>Clasificación:</strong> {classification}</p>
                <p><strong>Interpretación:</strong> {interpretation}</p>
            </div>

            <h2>Composición Proximal</h2>
            <table>
                <tr><th>Componente</th><th>Valor (%)</th></tr>
                <tr><td>Humedad</td><td>{data['moisture']:.2f}</td></tr>
                <tr><td>Proteína</td><td>{data['protein']:.2f}</td></tr>
                <tr><td>Grasa</td><td>{data['fat']:.2f}</td></tr>
                <tr><td>Cenizas</td><td>{data['ash']:.2f}</td></tr>
                <tr><td>Fibra cruda</td><td>{data['crude_fiber']:.2f}</td></tr>
                <tr><td>Carbohidratos totales</td><td>{data['total_carbohydrates']:.2f}</td></tr>
                <tr><td>Fibra dietética</td><td>{data['dietary_fiber']:.2f}</td></tr>
                <tr><td>Azúcares</td><td>{data['sugars']:.2f}</td></tr>
            </table>

            <h2>Explicabilidad del Modelo</h2>
            <h3>Análisis SHAP</h3>
            <img src="data:image/png;base64,{beeswarm_b64}" alt="SHAP Beeswarm">
            <p class="img-caption">Importancia global de las variables</p>

            <img src="data:image/png;base64,{waterfall_b64}" alt="SHAP Waterfall">
            <p class="img-caption">Contribución de variables para esta muestra</p>

            {f'<h2>Recomendaciones</h2><div class="highlight">{recommendations}</div>' if recommendations else ''}
        </body>
        </html>
        """

        # Generar PDF directamente en memoria
        pdf_buffer = BytesIO()
        pisa_status = pisa.CreatePDF(
            html_content,
            dest=pdf_buffer,
            encoding='UTF-8'
        )

        # Verificar errores
        if pisa_status.err:
            st.error("Error al generar el PDF")
            return None

        # Limpiar archivos temporales
        for img_file in [beeswarm_img, waterfall_img]:
            if os.path.exists(img_file):
                os.remove(img_file)

        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()

    except Exception as e:
        st.error(f"❌ Error al generar informe: {str(e)}")
        return None


def generate_batch_report_with_shap(df, waterfall_images):
    import os
    import base64
    from io import BytesIO
    from xhtml2pdf import pisa

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
        # --- Configuración inicial ---
        def img_to_base64(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        # --- Generar beeswarm plot ---
        input_batch = df[required_cols]
        explainer = get_shap_explainer()
        shap_values_batch = explainer(input_batch)
        shap_values_batch.feature_names = get_clean_feature_names()

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_values_batch, show=False)
        plt.tight_layout()
        beeswarm_path = "shap_beeswarm_lote.png"
        fig1.savefig(beeswarm_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close(fig1)
        beeswarm_b64 = img_to_base64(beeswarm_path)

        # --- Estadísticas ---
        avg_frap = df['FRAP_predicho'].mean()
        high_count = len(df[df['Clasificación'] == 'Alto'])
        med_count = len(df[df['Clasificación'] == 'Medio'])
        low_count = len(df[df['Clasificación'] == 'Bajo'])

        # --- Preparar HTML ---
        display_df = df.copy().rename(columns=COLUMN_TRANSLATIONS)
        df_html = display_df.to_html(index=False, classes="table", border=0)

        # --- Sección de recomendaciones ---
        recomendaciones_html = """
        <div style="page-break-before: always; text-align: justify;">
            <h2 style="text-align: center;">Recomendaciones para I+D</h2>
            <p>Clasificación basada en valores FRAP predichos:</p>
            <ul>
                <li><strong>Alto</strong>: > 40 mmol Fe²⁺/100g</li>
                <li><strong>Medio</strong>: 15-40 mmol Fe²⁺/100g</li>
                <li><strong>Bajo</strong>: < 15 mmol Fe²⁺/100g</li>
            </ul>
        """

        if high_count > 0:
            recomendaciones_html += f"""
            <h3 style="text-align: center;">Residuos con Alta Capacidad Antioxidante ({high_count})</h3>
            <p><strong>Estrategia:</strong> Priorizar extracción de compuestos bioactivos</p>
            <ul>
                <li>Desarrollo de extractos antioxidantes</li>
                <li>Aplicaciones en alimentos funcionales</li>
            </ul>
            """

        if med_count > 0:
            recomendaciones_html += f"""
            <h3 style="text-align: center;">Residuos con Capacidad Media ({med_count})</h3>
            <p><strong>Estrategia:</strong> Valorización integrada</p>
            <ul>
                <li>Uso como ingrediente complementario</li>
                <li>Procesos biotecnológicos</li>
            </ul>
            """

        if low_count > 0:
            recomendaciones_html += f"""
            <h3 style="text-align: center;">Residuos con Baja Capacidad ({low_count})</h3>
            <p><strong>Estrategia:</strong> Aprovechamiento estructural/energético</p>
            <ul>
                <li>Fuente de fibra para biomateriales</li>
                <li>Producción de biocombustibles</li>
            </ul>
            """

        recomendaciones_html += "</div>"

        # --- Waterfalls (2 por página) ---
        waterfalls_html = ""
        for i in range(0, len(waterfall_images), 2):
            img1, name1 = waterfall_images[i]
            img2, name2 = waterfall_images[i+1] if i+1 < len(waterfall_images) else (None, None)

            waterfalls_html += """
            <div style="page-break-before: always; display: flex; justify-content: space-between;">
                <div style="width: 48%;">
                    <h3 style="text-align: center;">📊 {}</h3>
                    <img src="data:image/png;base64,{}" style="max-width: 100%;">
                </div>
            """.format(name1, img_to_base64(img1))

            if img2:
                waterfalls_html += """
                <div style="width: 48%;">
                    <h3 style="text-align: center;">📊 {}</h3>
                    <img src="data:image/png;base64,{}" style="max-width: 100%;">
                </div>
                """.format(name2, img_to_base64(img2))

            waterfalls_html += "</div>"

        # --- HTML completo ---
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Informe por Lote</title>
            <style>
                @page {{ size: landscape; margin: 1cm; }}
                body {{ font-family: Arial; text-align: justify; }}
                h1, h2 {{ color: #2c3e50; text-align: center; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                .table th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Informe de Análisis por Lote</h1>
            <p><strong>Fecha:</strong> {datetime.now().strftime('%d/%m/%Y')}</p>
            <p><strong>Muestras:</strong> {len(df)} | <strong>FRAP promedio:</strong> {avg_frap:.2f}</p>

            <h2>Resultados</h2>
            {df_html}

            {recomendaciones_html}

            <h2>Análisis SHAP</h2>
            <h3>Importancia Global</h3>
            <img src="data:image/png;base64,{beeswarm_b64}">

            {waterfalls_html}
        </body>
        </html>
        """

        # --- Generar PDF en memoria ---
        pdf_data = BytesIO()
        pisa_status = pisa.CreatePDF(html_content, dest=pdf_data)

        if pisa_status.err:
            raise Exception("Error al generar PDF")

        # --- Limpieza ---
        for img_path, _ in waterfall_images:
            if os.path.exists(img_path):
                os.remove(img_path)
        if os.path.exists(beeswarm_path):
            os.remove(beeswarm_path)

        pdf_data.seek(0)
        return pdf_data.getvalue()

    except Exception as e:
        st.error(f"❌ Error al generar informe: {str(e)}")
        return None
