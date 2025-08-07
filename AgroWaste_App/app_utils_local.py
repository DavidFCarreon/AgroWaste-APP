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

    classification = "Alto" if frap_value > 40 else "Medio" if frap_value > 15 else "Bajo"
    interpretation = {"Alto": "Alto potencial funcional", "Medio": "Potencial moderado", "Bajo": "Bajo potencial"}[classification]
    recommendation = {"Alto": "Priorizar", "Medio": "Considerar", "Bajo": "Descartar"}[classification]

    try:
        def img_to_base64(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        beeswarm_b64 = img_to_base64(beeswarm_img)
        waterfall_b64 = img_to_base64(waterfall_img)

        # Preparar sección de recomendaciones si existen
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
                .recommendations {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-left: 4px solid #4CAF50;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
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


        with open("informe_temp.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        from weasyprint import HTML
        HTML("informe_temp.html").write_pdf("informe_con_shap.pdf")

        with open("informe_con_shap.pdf", "rb") as f:
            pdf_data = f.read()

        for file in ["informe_temp.html", "shap_beeswarm.png", "shap_waterfall.png", "informe_con_shap.pdf"]:
            if os.path.exists(file):
                os.remove(file)

        return pdf_data

    except Exception as e:
        st.error(f"❌ Error al generar informe: {e}")
        return None

# --- Función generate_batch_report_with_shap (modificada con orientación horizontal) ---
def generate_batch_report_with_shap(df, waterfall_images):
    import os
    import base64
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
        # Crear directorio temporal si no existe
        os.makedirs("temp_report", exist_ok=True)
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

        # Crear DataFrame para visualización con nombres traducidos
        display_df = df.copy()
        display_df = display_df.rename(columns=COLUMN_TRANSLATIONS)


        # Convertir DataFrame a HTML
        df_html = display_df.to_html(index=False, table_id="resultados", escape=False)

        # Generar secciones condicionales
        recomendaciones_html = """
        <div style="page-break-before: always;">
            <h2 style="text-align: center;">Recomendaciones para I+D</h2>
            <div style="text-align: justify; margin: 0 5%;">
                <p>Cuando se evalúa un conjunto diverso de residuos agroindustriales con fines de valorización, la clasificación de su actividad antioxidante, medida mediante el método FRAP, puede proporcionar un criterio útil para orientar de manera preliminar el enfoque tecnológico más adecuado a seguir. A continuación, se presenta una recomendación generalizada para cada nivel de clasificación:</p>
        """
        if high_count > 0:
            recomendaciones_html += f"""
                <h3 style="text-align: center;">Residuos con Alta Capacidad Antioxidante: {high_count}</h3>
                <p><strong style="text-align: center; display: block;">(FRAP > 10 mmol Fe²⁺/100 g)</strong></p>
                <p><strong>Estrategia sugerida:</strong> Priorizar la recuperación de compuestos bioactivos antioxidantes, como polifenoles, flavonoides o compuestos azufrados, mediante procesos de extracción optimizados.</p>
                <p><strong>Líneas recomendadas:</strong></p>
                <ul style="text-align: justify;">
                    <li>Extracción de compuestos fenólicos mediante tecnologías verdes (ultrasonido, microondas, fluidos supercríticos)</li>
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
                <p><strong>Estrategia sugerida:</strong> Considerar una valorización dual o integrada, combinando el aprovechamiento de compuestos bioactivos con otras fracciones funcionales del residuo.</p>
                <p><strong>Líneas recomendadas:</strong></p>
                <ul style="text-align: justify;">
                    <li>Desarrollo de extractos con funcionalidad moderada, aplicables como antioxidantes en matrices menos susceptibles a oxidación o en sinergia con otros aditivos.</li>
                    <li>Incorporación como ingrediente funcional con aporte antioxidante complementario, en alimentos, suplementos o fórmulas cosméticas.</li>
                    <li>Evaluación como fuente de fibra dietética u otros metabolitos secundarios.</li>
                    <li>Uso como sustrato en procesos biotecnológicos (fermentación, producción de enzimas o metabolitos de valor).</li>
                    <li>Aplicación en formulación de productos combinados con otros residuos que permitan sinergias funcionales.</li>
                </ul>
            """
        if low_count > 0:
            recomendaciones_html += f"""
                <h3 style="text-align: center;">Residuos con Baja Capacidad Antioxidante: {low_count}</h3>
                <p><strong style="text-align: center; display: block;">(FRAP < 2 mmol Fe²⁺/100 g)</strong></p>
                <p><strong>Estrategia sugerida:</strong> Desviar el enfoque hacia la valorización de otras fracciones estructurales o energéticas del residuo, dado que su contenido de compuestos antioxidantes no justifica una explotación orientada a bioactivos.</p>
                <p><strong>Líneas recomendadas:</strong></p>
                <ul style="text-align: justify;">
                    <li>Aprovechamiento como fuente de fibra estructural, celulosa, hemicelulosa o lignina para la elaboración de biomateriales (bioplásticos, papel, aditivos de construcción).</li>
                    <li>Producción de biocombustibles o bioenergía mediante hidrólisis y fermentación o digestión anaerobia.</li>
                    <li>Uso en alimentación animal, compostaje o formulación de enmiendas orgánicas, previa evaluación de composición nutricional y seguridad.</li>
                    <li>Aplicaciones en fermentación de estado sólido o líquido para obtención de subproductos industriales (enzimas, ácidos orgánicos, biopigmentos).</li>
                    <li>Considerar su inclusión como componente de mezclas multirresiduo, en esquemas de valorización combinada.</li>
                </ul>
            """
        recomendaciones_html += """
            </div>
        </div>
        """

        # Generar Waterfalls HTML con 2 gráficos por página (lado a lado)
        waterfalls_html = ""
        for i in range(0, len(waterfall_images), 2):
            # Primera imagen (siempre existe)
            img_path_1 = waterfall_images[i][0]
            sample_name_1 = waterfall_images[i][1]

            # Segunda imagen (si existe)
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
                    <!-- Figura 1 -->
                    <div style="width: 48%; margin-right: 2%;">
                        <h3>📊 SHAP Waterfall: {sample_name_1}</h3>
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{img_to_base64(img_path_1)}" alt="SHAP Waterfall" style="max-width: 95%; height: auto;">
                        </div>
                        <p class="img-caption">Desglose de la predicción para la muestra {sample_name_1}.</p>
                    </div>
                    <!-- Figura 2 -->
                    {second_image_html}
                </div>
            </div>
            """

        # HTML completo con ajustes mejorados para la tabla
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Informe por Lote - FRAP Predicho</title>
            <style>
                @page {{
                    size: landscape;
                    margin: 1cm; /* Reducir margen general */
                }}
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0 auto; /* Centrado horizontal */
                    max-width: 95%; /* Limitar ancho máximo */
                    padding: 0.5cm;
                    line-height: 1.4;
                    text-align: center; /* Centrado general */
                }}
                .content-wrapper {{
                    text-align: left; /* Alinear texto a izquierda dentro del contenedor */
                    margin: 0 auto; /* Centrado horizontal */
                    width: 100%;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                    text-align: center; /* Títulos centrados */
                    margin-top: 0.5em;
                    margin-bottom: 0.5em;
                }}
                h1 {{
                    font-size: 1.6em; /* Reducir tamaño para mejor ajuste */
                    page-break-before: avoid;
                }}
                h2 {{ font-size: 1.3em; }}
                h3 {{ font-size: 1.1em; }}
                .results-table-container {{
                    width: 100%;
                    overflow-x: auto;
                    margin: 15px auto; /* Centrado */
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    table-layout: auto;
                    font-size: 0.65em; /* Reducir ligeramente */
                    word-wrap: break-word;
                    margin: 0 auto; /* Centrar tabla */
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 5px 6px; /* Ajustar padding */
                    text-align: center; /* Contenido de celdas centrado */
                    vertical-align: middle;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                    white-space: nowrap;
                }}
                td {{
                    word-break: break-word;
                    max-width: 120px; /* Reducir ancho máximo */
                }}
                .highlight {{
                    background-color: #fffacd;
                    padding: 10px;
                    margin: 12px auto; /* Centrado */
                    border-radius: 5px;
                    width: 90%;
                    text-align: center;
                }}
                img {{
                    max-width: 85%; /* Reducir tamaño de imágenes */
                    height: auto;
                    margin: 8px auto; /* Centrado */
                    display: block;
                }}
                .img-caption {{
                    font-size: 0.8em;
                    color: #555;
                    text-align: center;
                    font-style: italic;
                    margin-top: 5px;
                }}
                ul {{
                    font-size: 0.85em;
                    line-height: 1.3;
                    padding-left: 20px;
                }}
                li {{
                    margin-bottom: 4px;
                    text-align: left;
                }}
                .waterfall-container {{
                    page-break-inside: avoid; /* Evitar división entre páginas */
                }}
                /* Control de paginación para recomendaciones */
                .recommendations-section {{
                    page-break-inside: avoid;
                    page-break-after: always;
                }}
            </style>
        </head>
        <body>
            <div class="content-wrapper">
                <h1>Informe de Análisis por Lote - FRAP Predicho</h1>
                <p style="text-align: center;"><strong>Fecha:</strong> {datetime.now().strftime('%d/%m/%Y')}</p>
                <p style="text-align: center;"><strong>Total de muestras:</strong> {len(df)}</p>
                <p style="text-align: center;"><strong>FRAP promedio:</strong> {avg_frap:.2f} mmol Fe²⁺/100g</p>
                <div class="highlight">
                    <p><strong>Resumen:</strong> Este informe evalúa el potencial antioxidante de múltiples agroresiduos.</p>
                </div>
                <h2>Resultados</h2>
                <div class="results-table-container">
                    {df_html}
                </div>
                <div class="recommendations-section">
                    {recomendaciones_html}
                </div>
                <div>
                    <h3>🐝 SHAP Beeswarm: Importancia global de features</h3>
                    <img src="data:image/png;base64,{beeswarm_b64}" alt="SHAP Beeswarm">
                    <p class="img-caption">Importancia de cada componente en el conjunto de predicciones.</p>
                </div>
                {waterfalls_html}
            </div>
        </body>
        </html>
        """

        # Guardar HTML temporal
        with open("lote_temp.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        # Generar PDF en modo paisaje
        from weasyprint import HTML
        HTML("lote_temp.html").write_pdf("informe_lote_con_shap.pdf")

        # Leer PDF generado
        with open("informe_lote_con_shap.pdf", "rb") as f:
            pdf_data = f.read()

        # Limpiar archivos temporales
        files_to_remove = [
            "lote_temp.html",
            "shap_beeswarm_lote.png",
            "informe_lote_con_shap.pdf"
        ]
        # Añadir solo las rutas de las imágenes (extraer de las tuplas)
        files_to_remove.extend([img_path for img_path, sample_name in waterfall_images])

        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)

        return pdf_data

    except Exception as e:
        st.error(f"❌ Error al generar informe por lote: {e}")
        return None
