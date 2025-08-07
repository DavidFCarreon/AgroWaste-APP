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
from fpdf import FPDF
import base64

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
X_background = df_background.iloc[:, :-1]
X_background_transformed = preprocessor.transform(X_background)
X_sampled_transformed = shap.sample(X_background_transformed, 100)
feature_names = X_background.columns.tolist()
X_sampled_df = pd.DataFrame(X_sampled_transformed, columns=feature_names)

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
            time.sleep(1)
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
            safe_name = f"sample_{idx}"
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
                    shap_values_list.append(result['shap_values'][0])
                    base_values_list.append(result['shap_base_values'][0])
                    feature_data_list.append(result['shap_data'][0])

                    explanation = shap.Explanation(
                        values=np.array([result['shap_values'][0]]),
                        base_values=np.array([result['shap_base_values'][0]]),
                        data=np.array([result['shap_data'][0]]),
                        feature_names=get_clean_feature_names()
                    )
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    shap.plots.waterfall(explanation[0], show=False)
                    plt.tight_layout()
                    img_path = f"shap_waterfall_{safe_name}.png"
                    fig.savefig(img_path, bbox_inches='tight', dpi=130, facecolor='white')
                    plt.close(fig)
                    waterfall_images.append((img_path, row['sample_name']))

                progress_bar.progress((idx + 1) / total_samples)
            except Exception as e:
                st.warning(f"Error procesando muestra {row['sample_name']}: {str(e)}")
                continue

    shap_values_global = shap.Explanation(
        values=np.array(shap_values_list),
        base_values=np.array(base_values_list),
        data=np.array(feature_data_list),
        feature_names=get_clean_feature_names()
    )
    return shap_values_global, waterfall_images

# --- Función generate_report_with_shap (usando fpdf2) ---
def generate_report_with_shap(data, frap_value, beeswarm_img, waterfall_img, recommendations=None):
    from fpdf import FPDF
    import os

    classification = "Alto" if frap_value > 40 else "Medio" if frap_value > 15 else "Bajo"
    interpretation = {"Alto": "Alto potencial funcional", "Medio": "Potencial moderado", "Bajo": "Bajo potencial"}[classification]
    recommendation = {"Alto": "Priorizar", "Medio": "Considerar", "Bajo": "Descartar"}[classification]

    try:
        # Crear PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Título
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Informe de Evaluación de Potencial Antioxidante", ln=True, align="C")
        pdf.ln(5)

        # Información básica
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Muestra: {data['sample_name']}", ln=True)
        pdf.cell(0, 8, f"Origen: {data.get('origin', 'N/A')}", ln=True)
        pdf.cell(0, 8, f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", ln=True)
        pdf.ln(5)

        # FRAP
        pdf.set_font("Arial", "B", 12)
        pdf.set_fill_color(230, 250, 230)
        pdf.cell(0, 10, f"FRAP predicho: {frap_value:.2f} mmol Fe²⁺/100g", ln=True, fill=True)
        pdf.cell(0, 10, f"Clasificación: {classification}", ln=True, fill=True)
        pdf.cell(0, 10, f"Interpretación: {interpretation}", ln=True, fill=True)
        pdf.cell(0, 10, f"Recomendación general: {recommendation}", ln=True, fill=True)
        pdf.ln(10)

        # Composición proximal
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Composición Proximal", ln=True)
        pdf.set_font("Arial", "B", 10)
        pdf.set_fill_color(240, 240, 240)
        headers = ["Componente", "Valor (%)"]
        col_widths = [60, 40]
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 8, h, 1, 0, "C", fill=True)
        pdf.ln(8)

        pdf.set_font("Arial", "", 10)
        proximal_data = [
            ("Humedad", data['moisture']),
            ("Proteína", data['protein']),
            ("Grasa", data['fat']),
            ("Ceniza", data['ash']),
            ("Fibra cruda", data['crude_fiber']),
            ("Carb. Totales", data['total_carbohydrates']),
            ("Fibra Dietética", data['dietary_fiber']),
            ("Azúcares", data['sugars']),
        ]
        for comp, val in proximal_data:
            pdf.cell(col_widths[0], 8, comp, 1, 0, "L")
            pdf.cell(col_widths[1], 8, str(val), 1, 0, "C")
            pdf.ln(8)
        pdf.ln(10)

        # SHAP Beeswarm
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "SHAP Beeswarm", ln=True)
        pdf.image(beeswarm_img, w=180)
        pdf.ln(5)

        # SHAP Waterfall
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "SHAP Waterfall", ln=True)
        pdf.image(waterfall_img, w=180)
        pdf.ln(5)

        # Recomendaciones
        if recommendations:
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Recomendaciones de Investigación y Desarrollo", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 6, f"Oportunidades de valorización para {data['sample_name']}:")
            pdf.ln(5)
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 6, recommendations)

        # Guardar PDF en bytes
        pdf_data = pdf.output(dest="S").encode("latin1")
        return pdf_data

    except Exception as e:
        print(f"❌ Error al generar informe individual: {e}")
        return None

# --- Función generate_batch_report_with_shap (usando fpdf2) ---
def generate_batch_report_with_shap(df, waterfall_images):
    from fpdf import FPDF
    import os

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
        input_batch = df[required_cols]
        explainer = get_shap_explainer()
        shap_values_batch = explainer(input_batch)
        shap_values_batch.feature_names = get_clean_feature_names()

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        shap.plots.beeswarm(shap_values_batch, show=False)
        plt.savefig("shap_beeswarm_lote.png", bbox_inches='tight', dpi=150, facecolor='white')
        plt.close(fig1)

        avg_frap = df['FRAP_predicho'].mean()
        high_count = len(df[df['Clasificación'] == 'Alto'])
        med_count = len(df[df['Clasificación'] == 'Medio'])
        low_count = len(df[df['Clasificación'] == 'Bajo'])

        display_df = df.rename(columns=COLUMN_TRANSLATIONS)

        # Crear PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Título
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Informe de Análisis por Lote - FRAP Predicho", ln=True, align="C")
        pdf.ln(5)

        # Información básica
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", ln=True)
        pdf.cell(0, 8, f"Total de muestras: {len(df)}", ln=True)
        pdf.cell(0, 8, f"FRAP promedio: {avg_frap:.2f} mmol Fe²⁺/100g", ln=True)
        pdf.ln(10)

        # Tabla de resultados
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Resultados", ln=True)
        pdf.set_font("Arial", "B", 8)
        col_widths = [40, 30, 15, 15, 15, 20, 20, 15, 20, 20]
        for col, width in zip(display_df.columns, col_widths):
            pdf.cell(width, 8, col, 1, 0, "C", fill=True)
        pdf.ln(8)

        pdf.set_font("Arial", "", 8)
        for _, row in display_df.iterrows():
            for col, width in zip(display_df.columns, col_widths):
                val = str(row[col])
                pdf.cell(width, 6, val[:12] + "..." if len(val) > 12 else val, 1, 0, "C")
            pdf.ln(6)
        pdf.ln(10)

        # Recomendaciones
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Recomendaciones para I+D", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 6, "Cuando se evalúa un conjunto diverso de residuos agroindustriales con fines de valorización...")

        if high_count > 0:
            pdf.ln(5)
            pdf.set_font("Arial", "B", 14)
            pdf.set_text_color(0, 100, 0)
            pdf.cell(0, 10, f"Residuos con Alta Capacidad Antioxidante: {high_count}", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", "I", 11)
            pdf.cell(0, 8, "(FRAP > 40 mmol Fe²⁺/100g)", ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 6, "Estrategia sugerida: Priorizar la recuperación de compuestos bioactivos antioxidantes...")
            pdf.set_font("Arial", "", 10)
            for item in [
                "Extracción de compuestos fenólicos mediante tecnologías verdes",
                "Desarrollo de ingredientes funcionales para alimentos y nutracéuticos",
                "Aplicaciones en cosmética natural como antioxidantes",
                "Microencapsulación para mejorar estabilidad y biodisponibilidad",
                "Evaluación sinérgica con otros antioxidantes naturales"
            ]:
                pdf.cell(10)
                pdf.cell(0, 6, f"• {item}", ln=True)
            pdf.ln(5)

        if med_count > 0:
            pdf.ln(5)
            pdf.set_font("Arial", "B", 14)
            pdf.set_text_color(0, 0, 139)
            pdf.cell(0, 10, f"Residuos con Capacidad Antioxidante Media: {med_count}", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", "I", 11)
            pdf.cell(0, 8, "(FRAP entre 15 y 40 mmol Fe²⁺/100g)", ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 6, "Estrategia sugerida: Considerar una valorización dual o integrada...")
            pdf.set_font("Arial", "", 10)
            for item in [
                "Desarrollo de extractos con funcionalidad moderada",
                "Incorporación como ingrediente funcional complementario",
                "Evaluación como fuente de fibra dietética u otros metabolitos",
                "Uso como sustrato en procesos biotecnológicos",
                "Aplicación en formulación de productos combinados"
            ]:
                pdf.cell(10)
                pdf.cell(0, 6, f"• {item}", ln=True)
            pdf.ln(5)

        if low_count > 0:
            pdf.ln(5)
            pdf.set_font("Arial", "B", 14)
            pdf.set_text_color(139, 0, 0)
            pdf.cell(0, 10, f"Residuos con Baja Capacidad Antioxidante: {low_count}", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", "I", 11)
            pdf.cell(0, 8, "(FRAP < 15 mmol Fe²⁺/100g)", ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 6, "Estrategia sugerida: Desviar el enfoque hacia otras fracciones...")
            pdf.set_font("Arial", "", 10)
            for item in [
                "Aprovechamiento como fuente de fibra estructural",
                "Producción de biocombustibles o bioenergía",
                "Uso en alimentación animal o compostaje",
                "Aplicaciones en fermentación de estado sólido o líquida",
                "Considerar su inclusión como componente de mezclas multirresiduo"
            ]:
                pdf.cell(10)
                pdf.cell(0, 6, f"• {item}", ln=True)
            pdf.ln(5)

        # SHAP Beeswarm
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "SHAP Beeswarm: Importancia global de features", ln=True)
        pdf.image("shap_beeswarm_lote.png", w=180)
        pdf.ln(10)

        # Waterfalls (2 por página)
        for i in range(0, len(waterfall_images), 2):
            pdf.add_page()
            img_path_1 = waterfall_images[i][0]
            sample_name_1 = waterfall_images[i][1]
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"SHAP Waterfall: {sample_name_1}", ln=True)
            pdf.image(img_path_1, w=180)
            pdf.ln(10)

            if i + 1 < len(waterfall_images):
                img_path_2 = waterfall_images[i+1][0]
                sample_name_2 = waterfall_images[i+1][1]
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"SHAP Waterfall: {sample_name_2}", ln=True)
                pdf.image(img_path_2, w=180)
                pdf.ln(10)

        # Guardar PDF en bytes
        pdf_data = pdf.output(dest="S").encode("latin1")

        # Limpiar imágenes
        files_to_remove = ["shap_beeswarm_lote.png"]
        files_to_remove.extend([img_path for img_path, _ in waterfall_images])
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)

        return pdf_data

    except Exception as e:
        print(f"❌ Error al generar informe por lote: {e}")
        return None
