# app.py (versión optimizada y compatible con Streamlit Cloud)
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import joblib
import json

# Importar funciones de app_utils
from app_utils import (
    get_clean_feature_names,
    safe_api_request,
    process_batch_shap,
    generate_report_with_shap,
    generate_batch_report_with_shap,
    get_background_data  # Ahora se importa explícitamente
)

# --- Carga de assets precalculados ---
@st.cache_resource
def load_shap_assets():
    try:
        assets = {
            'shap_values_global': joblib.load("AgroWaste_App/models/shap_values_global.joblib"),
            'feature_names': json.load(open("AgroWaste_App/models/feature_names.json")),
            'background_df': pd.read_pickle("AgroWaste_App/models/background_df.pkl")
        }
        return assets
    except Exception as e:
        st.error(f"Error cargando assets: {str(e)}")
        return None

# Cargamos los assets al inicio
shap_assets = load_shap_assets()

if shap_assets is None:
    st.stop()

# --- Configuración de funciones ---
get_clean_feature_names = lambda: shap_assets['feature_names']
get_background_data = lambda: shap_assets['background_df']

# --- Configuración de la página ---
st.set_page_config(
    page_title="AgroWaste-APP",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar base de datos de muestras
if 'samples_db' not in st.session_state:
    st.session_state.samples_db = []

# --- Título y descripción ---
st.title("🌱 AgroWaste-APP: Antioxidant Power Predictor")
st.markdown("""
**Aplicación para predecir actividad antioxidante (FRAP) a partir de la composición proximal de residuos agroindustriales**
""")

# --- Pestañas principales ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Predicción Individual",
    "Predicción por Lotes",
    "Búsqueda con IA",
    "Simulador",
    "Acerca de"
])

required_cols = [
    'moisture', 'protein', 'fat', 'ash',
    'crude_fiber', 'total_carbohydrates',
    'dietary_fiber', 'sugars'
]

# --- Función predict_row ---
def predict_row(row):
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
        url_pred = "https://agrowaste-app-476771143854.europe-west1.run.app/predict"
        response = safe_api_request(url_pred, params)
        prediction = response.json()
        frap = prediction.get("FRAP_value", 0)
        return float(np.round(frap, 2))
    except Exception as e:
        st.warning(f"Excepción para muestra {row.get('sample_name', '')}: {str(e)}")
        return None

# --- Tab 1: Predicción Individual ---
with tab1:
    st.header("Predicción Individual")
    col1, col2 = st.columns(2)
    sample_name = col1.text_input("Nombre del residuo agroindustrial", "Orujo de oliva")
    origin = col2.text_input("Origen del residuo (opcional)", "Ej: Oliva")

    cols = st.columns(4)
    Moisture = cols[0].number_input("Humedad", 0.0, 95.0, 6.8)
    Protein = cols[1].number_input("Proteína", 0.0, 50.0, 12.4)
    Fat = cols[2].number_input("Grasa", 0.0, 50.0, 5.1)
    Ash = cols[3].number_input("Cenizas", 0.0, 20.0, 2.9)

    cols2 = st.columns(4)
    Crude_Fiber = cols2[0].number_input("Fibra cruda", 0.0, 80.0, 25.3)
    Dietary_Fiber = cols2[1].number_input("Fibra dietética", 0.0, 80.0, 30.1)
    Total_Carbohydrates = cols2[2].number_input("Carbohid. Totales", 0.0, 90.0, 45.0)
    Sugars = cols2[3].number_input("Azúcares", 0.0, 50.0, 1.8)

    if st.button("Predecir FRAP"):
        row = {
            'sample_name': sample_name,
            'moisture': Moisture,
            'protein': Protein,
            'fat': Fat,
            'ash': Ash,
            'crude_fiber': Crude_Fiber,
            'total_carbohydrates': Total_Carbohydrates,
            'dietary_fiber': Dietary_Fiber,
            'sugars': Sugars
        }
        row_display = row.copy()
        row_display['origin'] = origin

        # Llamada a la API
        url_pred = "https://agrowaste-app-476771143854.europe-west1.run.app/predict"
        try:
            response1 = safe_api_request(url_pred, {
                'Moisture': Moisture,
                'Protein': Protein,
                'Fat': Fat,
                'Ash': Ash,
                'Crude_Fiber': Crude_Fiber,
                'Total_Carbohydrates': Total_Carbohydrates,
                'Dietary_Fiber': Dietary_Fiber,
                'Sugars': Sugars
            })
            prediction1 = response1.json()
            frap1 = prediction1.get("FRAP_value", 0)
            shap_values1 = np.array(prediction1["shap_values"])
            base_values1 = np.array(prediction1["shap_base_values"])
            feature_values1 = np.array(prediction1["shap_data"])

            # Mostrar resultado
            if frap1 < 15:
                st.warning(f"FRAP: **{frap1:.2f} mmol Fe²⁺/100g** - *Poder antioxidante bajo*")
            elif 15 <= frap1 < 40:
                st.info(f"FRAP: **{frap1:.2f} mmol Fe²⁺/100g** - *Poder antioxidante medio*")
            else:
                st.success(f"FRAP: **{frap1:.2f} mmol Fe²⁺/100g** - *Poder antioxidante alto*")

            # Recomendaciones dinámicas
            url_gc = "https://agrowaste-app-476771143854.europe-west1.run.app/get_comments"
            response_gc = safe_api_request(url_gc, {'FRAP_value': frap1, 'product_name': sample_name})
            comment = response_gc.json().get("Comments", "")

            st.markdown("""
            <h3 style='font-size: 24px; color: #2c3e50; margin-bottom: 10px;'>
                🔬 Recomendaciones de I+D para la muestra ingresada:
            </h3>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='font-size: 20px; line-height: 1.6; background-color: #f8f9fa;
                        padding: 15px; border-radius: 8px; border-left: 4px solid #4e79a7;text-align: justify'>
                {comment}
            </div>
            """, unsafe_allow_html=True)

            # SHAP Beeswarm global
            st.subheader("🐝 SHAP Beeswarm: Importancia global del modelo")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            shap.plots.beeswarm(shap_assets['shap_values_global'], show=False)
            st.pyplot(fig1)
            fig1.savefig("shap_beeswarm.png", bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig1)

            # SHAP Waterfall individual
            explanation = shap.Explanation(
                values=shap_values1,
                base_values=base_values1,
                data=feature_values1,
                feature_names=get_clean_feature_names()
            )
            st.subheader("📊 SHAP Waterfall")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(explanation[0], show=False)
            st.pyplot(fig2)
            fig2.savefig("shap_waterfall.png", bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig2)

            # Guardar muestra
            sample_to_save = row_display.copy()
            sample_to_save['source'] = "Predicción Individual"
            sample_to_save['id'] = f"ind_{datetime.now().strftime('%H%M%S')}"
            st.session_state.samples_db = [
                s for s in st.session_state.samples_db
                if s['sample_name'] != sample_to_save['sample_name'] or s['source'] != "Predicción Individual"
            ]
            st.session_state.samples_db.append(sample_to_save)

            # Generar PDF
            pdf_data = generate_report_with_shap(
                data=row_display,
                frap_value=frap1,
                beeswarm_img="shap_beeswarm.png",
                waterfall_img="shap_waterfall.png",
                recommendations=comment if comment else None
            )
            if pdf_data:
                st.download_button(
                    "📥 Descargar informe PDF con SHAP",
                    pdf_data,
                    f"informe_{sample_name}_con_shap.pdf",
                    "application/pdf"
                )
        except Exception as e:
            st.error(f"Error: {e}")

# --- Tab 2: Predicción por Lotes ---
with tab2:
    st.header("Carga de datos")
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv", help="El archivo debe contener las columnas requeridas")
    df = pd.read_csv(uploaded_file) if uploaded_file else None

    display_names = {
        'sample_name': 'Residuo agroindustrial',
        'origin': 'Origen',
        'moisture': 'Humedad',
        'protein': 'Proteína',
        'fat': 'Grasa',
        'ash': 'Ceniza',
        'crude_fiber': 'Fibra Cruda',
        'total_carbohydrates': 'Carbohidratos Totales',
        'dietary_fiber': 'Fibra Dietética',
        'sugars': 'Azúcares',
        'FRAP_predicho': 'FRAP Predicho (mmol Fe²⁺/100g)',
        'Clasificación': 'Clasificación'
    }

    if df is not None:
        st.header("📊 Análisis por lote")
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"🚫 Faltan columnas requeridas: {', '.join(missing)}")
        else:
            if 'sample_name' not in df.columns:
                df['sample_name'] = [f'Muestra {i+1}' for i in range(len(df))]

            url_pred = "https://agrowaste-app-476771143854.europe-west1.run.app/predict"
            with st.spinner("Realizando predicciones..."):
                df['FRAP_predicho'] = df.apply(predict_row, axis=1)
                df['Clasificación'] = df['FRAP_predicho'].apply(lambda x: "Alto" if x > 40 else "Medio" if x > 15 else "Bajo")

            display_df = df.rename(columns={k: v for k, v in display_names.items() if k in df.columns})
            st.markdown("### Resultados del análisis")
            st.markdown("""
            <style>
                div[data-testid="stDataFrame"] th { background-color: #f8f9fa; color: #2c3e50; text-align: center; }
                div[data-testid="stDataFrame"] td { text-align: center; }
            </style>
            """, unsafe_allow_html=True)
            st.dataframe(
                display_df.sort_values("FRAP Predicho (mmol Fe²⁺/100g)", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            # Gráfico de barras
            palette = {'Alto': "#7ADAA5", 'Medio': "#239BA7", 'Bajo': '#E1AA36'}
            fig, ax = plt.subplots(figsize=(13, 7))
            barplot = sns.barplot(
                data=df,
                x='sample_name',
                y='FRAP_predicho',
                hue='Clasificación',
                palette=palette,
                dodge=False,
                ax=ax
            )
            ax.set_title("Potencial Antioxidante FRAP por Muestra", fontsize=15, pad=20, color='#2c3e50')
            ax.set_ylabel(r"FRAP$_{\text{pred}}$ (mmol Fe$^{2+}$/100g)")
            ax.set_xlabel("Muestras Analizadas")
            plt.xticks(rotation=45, ha='right')
            for p in barplot.patches:
                if p.get_height() > 0.01:
                    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=10, fontweight='bold', color="#0f161d")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.download_button("📥 Resultados CSV", df.to_csv(index=False), "resultados.csv", "text/csv")

            # Recomendaciones
            high_count = len(df[df['Clasificación'] == 'Alto'])
            med_count = len(df[df['Clasificación'] == 'Medio'])
            low_count = len(df[df['Clasificación'] == 'Bajo'])

            st.markdown("""
            <div class="recommendation-container">
                <h3 class="recommendation-title">
                    🔬 Recomendaciones de I+D para el lote de muestras:
                </h3>
            """, unsafe_allow_html=True)

            if high_count > 0:
                st.markdown(f"""
                <div class="recommendation-text">
                    <h4 class="recommendation-subtitle">Residuos con Alta Capacidad Antioxidante: {high_count}</h4>
                    <p><strong>Estrategia sugerida:</strong> Priorizar la recuperación de compuestos bioactivos antioxidantes.</p>
                    <ul class="recommendation-list">
                        <li>Extracción de compuestos fenólicos mediante tecnologías verdes</li>
                        <li>Desarrollo de ingredientes funcionales para alimentos y nutracéuticos</li>
                        <li>Aplicaciones en cosmética natural como antioxidantes</li>
                        <li>Microencapsulación para mejorar estabilidad y biodisponibilidad</li>
                        <li>Evaluación sinérgica con otros antioxidantes naturales</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            if med_count > 0:
                st.markdown(f"""
                <div class="recommendation-text">
                    <h4 class="recommendation-subtitle">Residuos con Capacidad Antioxidante Media: {med_count}</h4>
                    <p><strong>Estrategia sugerida:</strong> Considerar una valorización dual o integrada.</p>
                    <ul class="recommendation-list">
                        <li>Desarrollo de extractos con funcionalidad moderada</li>
                        <li>Incorporación como ingrediente funcional complementario</li>
                        <li>Evaluación como fuente de fibra dietética u otros metabolitos</li>
                        <li>Uso como sustrato en procesos biotecnológicos</li>
                        <li>Aplicación en formulación de productos combinados</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            if low_count > 0:
                st.markdown(f"""
                <div class="recommendation-text">
                    <h4 class="recommendation-subtitle">Residuos con Baja Capacidad Antioxidante: {low_count}</h4>
                    <p><strong>Estrategia sugerida:</strong> Desviar el enfoque hacia otras fracciones.</p>
                    <ul class="recommendation-list">
                        <li>Aprovechamiento como fuente de fibra estructural</li>
                        <li>Producción de biocombustibles o bioenergía</li>
                        <li>Uso en alimentación animal o compostaje</li>
                        <li>Aplicaciones en fermentación de estado sólido o líquida</li>
                        <li>Considerar su inclusión como componente de mezclas multirresiduo</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # SHAP por lote
            shap_values_global, waterfall_data = process_batch_shap(df, url_pred)

            st.subheader("🐝 SHAP Beeswarm (Importancia global)")
            fig_bee, ax_bee = plt.subplots(figsize=(10, 6))
            shap.plots.beeswarm(shap_values_global, show=False)
            st.pyplot(fig_bee)

            st.subheader("📊 SHAP Waterfall por muestra")
            cols = st.columns(2)
            for i, (img_path, sample_name) in enumerate(waterfall_data):
                with cols[i % 2]:
                    st.image(img_path, caption=f"Muestra: {sample_name}")
                    with open(img_path, "rb") as f:
                        img_data = f.read()
                    st.download_button(f"Descargar {sample_name}", img_data, f"shap_{sample_name}.png", "image/png")

            # Generar informe por lote
            pdf_data = generate_batch_report_with_shap(df, waterfall_data)
            if pdf_data:
                st.download_button(
                    "📥 Descargar informe por lote con SHAP",
                    pdf_data,
                    "informe_lote_con_shap.pdf",
                    "application/pdf"
                )

            # Limpieza
            for img_path, _ in waterfall_data:
                if os.path.exists(img_path):
                    os.remove(img_path)

# --- Tab 3: Búsqueda con IA ---
with tab3:
    st.header("Búsqueda con IA (Open AI GPT-4 mini)")
    search_gpt = st.text_input("Ingresa el nombre de un residuo agroindustrial:", "Bagazo de manzana")
    if st.button("Predecir FRAP con IA"):
        url_search = "https://agrowaste-app-476771143854.europe-west1.run.app/get_features"
        try:
            s_response = safe_api_request(url_search, {'product_name': search_gpt})
            prediction2 = s_response.json()
            # Extraer datos
            moist2 = prediction2.get("Moisture", 0)
            prot2 = prediction2.get("Protein", 0)
            fat2 = prediction2.get("Fat", 0)
            tot_carb2 = prediction2.get("Total_Carbohydrates", 0)
            sugars2 = prediction2.get("Sugars", 0)
            diet_fiber2 = prediction2.get("Dietary_Fiber", 0)
            crude_fiber2 = prediction2.get("Crude_Fiber", 0)
            ash2 = prediction2.get("Ash", 0)
            frap2 = prediction2.get("FRAP_value", 0)
            shap_values2 = np.array(prediction2["shap_values"])
            base_values2 = np.array(prediction2["shap_base_values"])
            feature_values2 = np.array(prediction2["shap_data"])

            if frap2 < 15:
                st.warning(f"FRAP: **{frap2:.2f} mmol Fe²⁺/100g** - *Poder antioxidante bajo*")
            elif 15 <= frap2 < 40:
                st.info(f"FRAP: **{frap2:.2f} mmol Fe²⁺/100g** - *Poder antioxidante medio*")
            else:
                st.success(f"FRAP: **{frap2:.2f} mmol Fe²⁺/100g** - *Poder antioxidante alto*")

            row_display2 = {
                'sample_name': search_gpt,
                'origin': search_gpt,
                'moisture': moist2,
                'protein': prot2,
                'fat': fat2,
                'ash': ash2,
                'crude_fiber': crude_fiber2,
                'total_carbohydrates': tot_carb2,
                'dietary_fiber': diet_fiber2,
                'sugars': sugars2
            }

            # Guardar en DB
            sample_to_save_ia = row_display2.copy()
            sample_to_save_ia['source'] = "Búsqueda con IA"
            sample_to_save_ia['id'] = f"ia_{datetime.now().strftime('%H%M%S')}"
            st.session_state.samples_db = [
                s for s in st.session_state.samples_db
                if s['sample_name'] != sample_to_save_ia['sample_name'] or s['source'] != "Búsqueda con IA"
            ]
            st.session_state.samples_db.append(sample_to_save_ia)

            # Mostrar tabla
            proximal = {
                "Componente": ["Humedad", "Proteína", "Grasa", "Carbohidratos Totales",
                               "Azúcares", "Fibra Dietética", "Fibra Cruda", "Cenizas", "FRAP"],
                "Valor": [moist2, prot2, fat2, tot_carb2, sugars2, diet_fiber2, crude_fiber2, ash2, round(frap2, 2)],
                "Descripción": [
                    "Contenido de agua en el residuo (g/100g)",
                    "Contenido proteico total (g/100g)",
                    "Contenido lipídico total (g/100g)",
                    "Suma de todos los carbohidratos (g/100g)",
                    "Carbohidratos de cadena corta (g/100g)",
                    "Fibra beneficiosa para la digestión (g/100g)",
                    "Fibra indigerible (g/100g)",
                    "Contenido mineral inorgánico (g/100g)",
                    "Capacidad antioxidante mmol (Fe2+/100g)"
                ]
            }
            st.dataframe(proximal, hide_index=True, use_container_width=True)

            # Recomendaciones
            url_gc2 = "https://agrowaste-app-476771143854.europe-west1.run.app/get_comments"
            response_gc2 = safe_api_request(url_gc2, {'FRAP_value': frap2, 'product_name': search_gpt})
            comment2 = response_gc2.json().get("Comments", "")
            st.markdown("""
                <h3 style='font-size: 24px; color: #2c3e50; margin-bottom: 10px;'>🔬 Recomendaciones de I+D para la muestra ingresada:</h3>
                """, unsafe_allow_html=True)
            st.markdown(f"""
                <div style='font-size: 18px; line-height: 1.7; background-color: #f8f9fa; padding: 18px; border-radius: 8px;
                            border-left: 4px solid #4e79a7; text-align: justify; margin-bottom: 20px;'>
                    {comment2}
                </div>
                """, unsafe_allow_html=True)

            # SHAP
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            shap.plots.beeswarm(shap_assets['shap_values_global'], show=False)
            st.pyplot(fig1)
            fig1.savefig("shap_beeswarm.png", bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig1)

            explanation = shap.Explanation(
                values=shap_values2,
                base_values=base_values2,
                data=feature_values2,
                feature_names=get_clean_feature_names()
            )
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(explanation[0], show=False)
            st.pyplot(fig2)
            fig2.savefig("shap_waterfall.png", bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig2)

            # Generar PDF
            pdf_data = generate_report_with_shap(
                data=row_display2,
                frap_value=frap2,
                beeswarm_img="shap_beeswarm.png",
                waterfall_img="shap_waterfall.png",
                recommendations=comment2
            )
            if pdf_data:
                st.download_button(
                    "📥 Descargar informe PDF con SHAP",
                    pdf_data,
                    f"informe_{search_gpt}_con_shap.pdf",
                    "application/pdf"
                )
        except Exception as e:
            st.error(f"Error: {e}")

# --- Tab 4: Simulador ---
with tab4:
    st.header("Simulador")
    st.markdown("Explora cómo cambios en la composición afectan el FRAP.")
    with st.expander("What-if"):
        traduccion_componentes = {
            'moisture': 'Humedad', 'protein': 'Proteína', 'fat': 'Grasa', 'ash': 'Ceniza',
            'crude_fiber': 'Fibra Cruda', 'total_carbohydrates': 'Carb. Totales',
            'dietary_fiber': 'Fibra Diet.', 'sugars': 'Azúcares'
        }
        opciones = []
        datos_opciones = []

        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                if row['sample_name'] not in opciones:
                    opciones.append(row['sample_name'])
                    datos_opciones.append(row.to_dict())

        for s in st.session_state.samples_db:
            if s['sample_name'] not in opciones:
                opciones.append(s['sample_name'])
                datos_opciones.append(s.copy())

        try:
            current_manual_name = sample_name or "Muestra Manual"
            if current_manual_name not in opciones:
                muestra_manual = {
                    'sample_name': current_manual_name,
                    'moisture': Moisture, 'protein': Protein, 'fat': Fat, 'ash': Ash,
                    'crude_fiber': Crude_Fiber, 'total_carbohydrates': Total_Carbohydrates,
                    'dietary_fiber': Dietary_Fiber, 'sugars': Sugars
                }
                opciones.append(current_manual_name)
                datos_opciones.append(muestra_manual)
        except:
            pass

        if len(opciones) == 0:
            st.info("No hay muestras disponibles.")
        else:
            base_sample_name = st.selectbox("Selecciona muestra", opciones)
            idx = opciones.index(base_sample_name)
            base_row = datos_opciones[idx]

            componente_seleccionado = st.selectbox("Modificar componente", options=required_cols,
                                                  format_func=lambda x: traduccion_componentes[x])
            change = st.slider(f"Δ {traduccion_componentes[componente_seleccionado]} (%)", -20.0, 20.0, 0.0, 0.5)
            modified = base_row.copy()
            modified[componente_seleccionado] += change
            modified[componente_seleccionado] = max(0.0, modified[componente_seleccionado])

            def get_frap_prediction(data):
                params_sim = {
                    'Moisture': data['moisture'], 'Protein': data['protein'], 'Fat': data['fat'],
                    'Ash': data['ash'], 'Crude_Fiber': data['crude_fiber'],
                    'Total_Carbohydrates': data['total_carbohydrates'],
                    'Dietary_Fiber': data['dietary_fiber'], 'Sugars': data['sugars']
                }
                try:
                    url_pred = "https://agrowaste-app-476771143854.europe-west1.run.app/predict"
                    response_sim = safe_api_request(url_pred, params_sim)
                    return response_sim.json().get("FRAP_value", 0)
                except:
                    return None

            with st.spinner("Calculando..."):
                frap_orig = get_frap_prediction(base_row)
                frap_mod = get_frap_prediction(modified)

            if frap_orig is not None and frap_mod is not None:
                st.metric("Original", f"{frap_orig:.2f}")
                st.metric("Modificado", f"{frap_mod:.2f}", delta=f"{frap_mod - frap_orig:+.2f}")
                if frap_mod > frap_orig:
                    st.success("✅ El cambio aumentaría el potencial antioxidante.")
                elif frap_mod < frap_orig:
                    st.warning("⚠️ El cambio reduciría el potencial.")
                else:
                    st.info("➡️ Sin cambio significativo.")
            else:
                st.error("Error al obtener predicciones.")

# --- Tab 5: Acerca de ---
with tab5:
    st.markdown("""
    ### 📌 Información General
    - **Versión:** 1.0
    - **Última actualización:** 06/08/2025
    - **Desarrollado por:** Itzel, Nancy, Julio, David
    - **Repositorio:** https://github.com/DavidFCarreon/AgroWaste-APP
    """)
    with st.expander("⚠️ Limitaciones del Modelo"):
        st.markdown("""
        - R²: 0.68
        - Las predicciones son estimaciones. Se recomienda validación experimental.
        """)
    with st.expander("📚 Guía de Uso"):
        st.markdown("""
        1. **Predicción Individual**: Ingrese valores manualmente
        2. **Predicción por Lotes**: Suba un archivo CSV
        3. **Simulador**: Explore escenarios
        """)
    with st.expander("Tecnologías utilizadas"):
        st.markdown("""
        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
            <img src="https://img.shields.io/badge/Python-3776AB?logo=python" alt="Python">
            <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit" alt="Streamlit">
            <img src="https://img.shields.io/badge/SHAP-FF6D01?logo=shap" alt="SHAP">
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Esta aplicación es para fines de investigación.")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📚 Guía Rápida")
    st.markdown("""
    1. **Predicción Individual**: Valores manuales
    2. **Predicción por Lotes**: CSV
    3. **Simulador**: Cambios hipotéticos
    """)
    st.markdown("### 🔍 Método FRAP")
    st.markdown("Mide capacidad de reducir Fe³⁺ a Fe²⁺.")
    st.markdown("### 📊 Clasificación FRAP")
    st.markdown("""
    - **Alto**: > 40
    - **Medio**: 15-40
    - **Bajo**: < 15
    """)
    try:
        with open("AgroWaste_App/dataset/Ejemplo.csv", "r") as f:
            csv_example = f.read()
        st.sidebar.download_button("📥 Ejemplo CSV", csv_example, "ejemplo_datos.csv", "text/csv")
    except:
        pass
