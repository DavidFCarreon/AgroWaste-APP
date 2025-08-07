# app.py (versión optimizada)
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime
import os
import joblib
import json
from jinja2 import Template
from app_utils import process_batch_shap, generate_report_with_shap , generate_batch_report_with_shap

# --- Carga de assets precalculados ---
@st.cache_resource
def load_shap_assets():
    """Carga los valores SHAP y metadatos precalculados"""
    assets = {
        'shap_values_global': joblib.load("AgroWaste_App/models/shap_values_global.joblib"),
        'feature_names': json.load(open("AgroWaste_App/models/feature_names.json")),
        'background_df': pd.read_pickle("AgroWaste_App/models/background_df.pkl")
    }
    return assets

# --- Función predict_row  ---
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
        response = requests.get(url_pred, params=params)
        if response.status_code == 200:
            prediction = response.json()
            frap = prediction.get("FRAP_value", 0)
            return frap
        else:
            st.warning(f"Error para muestra {row.get('sample_name', '')}: {response.text}")
            return None
    except Exception as e:
        st.warning(f"Excepción para muestra {row.get('sample_name', '')}: {str(e)}")
        return None

# Cargamos los assets al inicio
shap_assets = load_shap_assets()

try:
    from app_utils import get_shap_explainer
    # Modificación: Ahora get_clean_feature_names y get_background_data vienen de los assets
    get_clean_feature_names = lambda: shap_assets['feature_names']
    get_background_data = lambda: shap_assets['background_df']

except ImportError as e:
    st.error("❌ No se pudo cargar modelo_frap.py"); st.stop()

# --- Configuración original de la página ---
st.set_page_config(
    page_title="FRAP Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título y descripción ---
st.title("🌱 AgroWaste-APP: Antioxidant Power Predictor")
st.markdown("""
**Aplicación para predecir actividad antioxidante (FRAP) a partir de la composición proximal de residuos agroindustriales**
""")

# --- Pestañas principales ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Predicción Individual", "Predicción por Lotes", "Simulador", "Búsqueda con IA", "Acerca de"])

required_cols = ['moisture','protein','fat','ash','crude_fiber','total_carbohydrates','dietary_fiber','sugars']

# --- tab1: Predicción Individual (optimizada) ---
with tab1:
    st.header("Predicción Individual")
    col1, col2 = st.columns(2)
    sample_name = col1.text_input("Nombre del residuo agroindustrial", "Ingresar para obtener recomendaciones dinámicas")
    origin = col2.text_input("Origen", "Origen del residuo (opcional)")
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
        row_display['sample_name'] = sample_name
        row_display['origin'] = origin

        params = {
            'Moisture': Moisture,
            'Protein': Protein,
            'Fat': Fat,
            'Ash': Ash,
            'Crude_Fiber': Crude_Fiber,
            'Total_Carbohydrates': Total_Carbohydrates,
            'Dietary_Fiber': Dietary_Fiber,
            'Sugars': Sugars
        }

        # Llamar a la API
        url_pred = "https://agrowaste-app-476771143854.europe-west1.run.app/predict"

        try:
            response = requests.get(url_pred, params=params)
            if response.status_code == 200:
                prediction = response.json()
                frap = prediction.get("FRAP_value", 0)
                shap_values = np.array(prediction["shap_values"])
                base_values = np.array(prediction["shap_base_values"])
                feature_values = np.array(prediction["shap_data"])
                # Mostrar el resultado con estilo condicional
                if frap < 15:
                    st.warning(f"FRAP: **{frap:.2f} mmol Fe²⁺/100g** - *Poder antioxidante bajo*")
                elif 15 <= frap < 40:
                    st.info(f"FRAP: **{frap:.2f} mmol Fe²⁺/100g** - *Poder antioxidante medio*")
                else:  # frap >= 40
                    st.success(f"FRAP: **{frap:.2f} mmol Fe²⁺/100g** - *Poder antioxidante alto*")
            else:
                st.error(f"Error en la API: {response.status_code} - {response.text}")

            # === Recomendaciones dinámicas ===
            params_gc = {
                'FRAP_value': frap,
                'product_name': sample_name,
            }
            url_gc = "https://agrowaste-app-476771143854.europe-west1.run.app/get_comments"

            response_gc = requests.get(url_gc, params=params_gc)

            if response_gc.status_code == 200:
                recom = response_gc.json()
                comment = recom.get("Comments", 0)
                st.subheader("🔬 Recomendaciones de I+D para la muestra ingresada:")
                st.markdown(f"{comment}")
            else:
                st.error(f"Error en la API. No fue posible obtener recomendaciones dinámicas")
                pass

            # === SHAP ===
            st.subheader("🐝 SHAP Beeswarm: Importancia global del modelo")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            shap.plots.beeswarm(shap_assets['shap_values_global'], show=False)
            st.pyplot(fig1)
            fig1.savefig("shap_beeswarm.png", bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig1)

            # === Waterfall para la muestra actual (se calcula solo esto) ===
            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_values,
                data=feature_values,
                feature_names=["Humedad", "Proteína", "Grasa", "Ceniza", "Fibra Cruda",
                            "Carb. Totales", "Fibra Dietética", "Azúcares"]  # Ajusta los nombres
            )
            st.subheader("📊 SHAP Waterfall")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(explanation[0], show=False)
            st.pyplot(fig2)
            fig2.savefig("shap_waterfall.png", bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig2)

            # Generar informe
            pdf_data = generate_report_with_shap(
                data=row_display,
                frap_value=frap,
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



# --- tab2: Predicción por Lotes (optimizada) ---
with tab2:
    # --- Carga de datos ---
    st.header("Carga de datos")
    uploaded_file = st.file_uploader("Sube CSV", type="csv")
    df = pd.read_csv(uploaded_file) if uploaded_file else None

    # Mapeo de nombres de columnas (si es necesario)
    column_mapping = {
        'moisture': 'Moisture',
        'protein': 'Protein',
        'fat': 'Fat',
        'ash': 'Ash',
        'crude_fiber': 'Crude_Fiber',
        'total_carbohydrates': 'Total_Carbohydrates',
        'dietary_fiber': 'Dietary_Fiber',
        'sugars': 'Sugars'
    }

    try:
        feature_names_clean = get_clean_feature_names()
    except:
        feature_names_clean = [
            "Moisture", "Protein", "Fat", "Ash",
            "Crude Fiber", "Total Carbohydrates",
            "Dietary Fiber", "Sugars"
        ]

    if df is not None:
        st.header("Análisis por lote")
        df = df.rename(columns={v: k for k, v in column_mapping.items() if v in df.columns})

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Faltan columnas requeridas: {missing}")
        else:
            url_pred = "https://agrowaste-app-476771143854.europe-west1.run.app/predict"

            with st.spinner("Realizando predicciones..."):
                df['FRAP_predicho'] = df.apply(predict_row, axis=1)
                df['Clasificación'] = df['FRAP_predicho'].apply(lambda x: "Alto" if x > 40 else "Medio" if x > 15 else "Bajo")

            # Mostrar resultados
            st.dataframe(df.sort_values("FRAP_predicho", ascending=False))

            # Gráfica de barras
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                data=df,
                x='sample_name',
                y='FRAP_predicho',
                hue='Clasificación',
                dodge=False,
                ax=ax)
            plt.xticks(rotation=45)
            ax.set_title("Predicción de FRAP")
            plt.ylabel(r"FRAP$_{\text{pred}}$ (mmol Fe$^{2+}$/100g)")
            plt.xlabel("Muestra")
            st.pyplot(fig)

            # Botón para descargar CSV
            @st.cache_data
            def to_csv(x):
                return x.to_csv(index=False).encode('utf-8')

            st.download_button("📥 Resultados CSV",
                               to_csv(df),
                               "resultados.csv",
                               "text/csv")

            # Procesamiento SHAP
            #if st.button("🔄 Generar análisis SHAP"):
            shap_values_global, waterfall_data = process_batch_shap(df, url_pred)

            # === 1. SHAP Beeswarm para todo el lote ===
            st.subheader("🐝 SHAP Beeswarm (Importancia global)")
            fig_bee, ax_bee = plt.subplots(figsize=(10, 6))
            shap.plots.beeswarm(shap_values_global, show=False)
            st.pyplot(fig_bee)

            # === 2. Generar Waterfall individual para CADA muestra ===
            # Mostrar waterfalls individuales
            st.subheader("📊 SHAP Waterfall por muestra")
            cols = st.columns(2)  # 2 columnas para organizar los gráficos
            for i, (img_path, sample_name) in enumerate(waterfall_data):  # Desempaquetamos la tupla aquí
                with cols[i % 2]:  # Alternar entre columnas
                    st.image(img_path, caption=f"Muestra: {sample_name}")  # Usamos sample_name directamente

                    # Opcional: Botón para descargar cada imagen
                    with open(img_path, "rb") as f:
                        img_data = f.read()
                    st.download_button(
                        f"Descargar {sample_name}",
                        img_data,
                        f"shap_waterfall_{sample_name}.png",
                        "image/png"
                    )

            # === 3. Generar informe PDF por lote con todos los Waterfalls ===
            if st.button("Generar informe PDF por lote con SHAP"):
                pdf_data = generate_batch_report_with_shap(df, waterfall_data)
                if pdf_data:
                    st.download_button(
                        "📥 Descargar informe por lote con SHAP",
                        pdf_data,
                        "informe_lote_con_shap.pdf",
                        "application/pdf"
                    )

            # Limpieza de archivos temporales
            for img_tuple in waterfall_data:  # img_tuple es (img_path, sample_name)
                img_path = img_tuple[0]  # Extraemos solo la ruta
                if os.path.exists(img_path):
                    os.remove(img_path)


# --- Simulador "What-if" ---
with tab3:
    st.header("Simulador")
    with st.expander("What-if"):

        # Crear lista de opciones: muestra manual + muestras por lote
        opciones = []
        datos_opciones = []

        # Añadir muestra manual si fue ingresada
        if 'moisture' in locals():
            muestra_manual = {
                'sample_name': sample_name or "Muestra Manual",
                'moisture': Moisture ,
                'protein': Protein ,
                'fat': Fat ,
                'ash': Ash ,
                'crude_fiber': Crude_Fiber,
                'total_carbohydrates': Total_Carbohydrates,
                'dietary_fiber': Dietary_Fiber,
                'sugars': Sugars
            }
            opciones.append(muestra_manual['sample_name'])
            datos_opciones.append(muestra_manual)

        # Añadir muestras por lote
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                opciones.append(row['sample_name'])
                datos_opciones.append(row.to_dict())

        if len(opciones) == 0:
            st.info("No hay muestras disponibles para simular.")
        else:
            base_sample_name = st.selectbox("Selecciona muestra", opciones)
            idx = opciones.index(base_sample_name)
            base_row = datos_opciones[idx]

            component = st.selectbox("Modificar componente", required_cols)
            change = st.slider(f"Δ {component} (%)", -20.0, 20.0, 0.0, 0.5)

            modified = base_row.copy()
            modified[component] += change
            modified[component] = max(0.0, modified[component])

            # Función para obtener predicción desde API
            def get_frap_prediction(data):
                params = {
                    'Moisture': data['moisture'],
                    'Protein': data['protein'],
                    'Fat': data['fat'],
                    'Ash': data['ash'],
                    'Crude_Fiber': data['crude_fiber'],
                    'Total_Carbohydrates': data['total_carbohydrates'],
                    'Dietary_Fiber': data['dietary_fiber'],
                    'Sugars': data['sugars']
                }
                try:
                    response = requests.get("https://agrowaste-app-476771143854.europe-west1.run.app/predict", params=params)
                    if response.status_code == 200:
                        return response.json().get("FRAP_value", 0)
                    return None
                except Exception:
                    return None

            # Obtener predicciones
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
                st.error("Error al obtener predicciones. Verifica la conexión con la API.")


# --- Integración con OpenAI ---
with tab4:
    st.header("Búsqueda con IA (GPT)")
    search_gpt = st.text_input("Ingresa el nombre de un residuo agroindustrial:", "")
    if search_gpt:
        url_search = "https://agrowaste-app-476771143854.europe-west1.run.app/get_features"
        try:
            s_response = requests.get(url_search, params={'product_name': search_gpt})
            if s_response.status_code == 200:
                prediction2 = s_response.json()

                # Extraer valores
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

                # Crear tabla de resultados
                st.markdown("### Resultados encontrados")
                # Mostrar el resultado con estilo condicional
                if frap2 < 15:
                    st.warning(f"FRAP: **{frap2:.2f} mmol Fe²⁺/100g** - *Poder antioxidante bajo*")
                elif 15 <= frap2 < 40:
                    st.info(f"FRAP: **{frap2:.2f} mmol Fe²⁺/100g** - *Poder antioxidante medio*")
                else:  # frap2 >= 40
                    st.success(f"FRAP: **{frap2:.2f} mmol Fe²⁺/100g** - *Poder antioxidante alto*")

                row2 = {
                    'moisture': moist2,
                    'protein': prot2,
                    'fat': fat2,
                    'ash': ash2,
                    'crude_fiber': crude_fiber2,
                    'total_carbohydrates': tot_carb2,
                    'dietary_fiber': diet_fiber2,
                    'sugars': sugars2
                    }

                row_display2 = row2.copy()
                row_display2['sample_name'] = search_gpt
                row_display2['origin'] = search_gpt

                # Mostrar tabla con los valores proximales
                proximal = {
                    "Componente": ["Humedad", "Proteína", "Grasa", "Carbohidratos Totales",
                                 "Azúcares", "Fibra Dietética", "Fibra Cruda", "Cenizas", "FRAP"],
                    "Valor": [moist2, prot2, fat2,
                             tot_carb2, sugars2, diet_fiber2,
                             crude_fiber2, ash2, round(frap2, 2)],
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

                # Mostrar la tabla con estilo
                st.markdown("### Resumen de resultados:")
                st.dataframe(
                    proximal,
                    column_config={
                        "Componente": "Componente",
                        "Valor": st.column_config.NumberColumn("Valor"),
                        "Descripción": "Descripción"
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.error(f"Error en la API: {s_response.status_code} - {s_response.text}")

            # === Recomendaciones dinámicas ===
            params_gc2 = {
                'FRAP_value': frap2,
                'product_name': search_gpt,
            }
            url_gc2 = "https://agrowaste-app-476771143854.europe-west1.run.app/get_comments"

            response_gc2 = requests.get(url_gc2, params=params_gc2)

            if response_gc2.status_code == 200:
                recom2 = response_gc2.json()
                comment2 = recom2.get("Comments", 0)
                st.subheader("🔬 Recomendaciones de I+D para la muestra ingresada:")
                st.markdown(f"{comment2}")
            else:
                st.error(f"Error en la API. No fue posible obtener recomendaciones dinámicas")
                pass


            # === SHAP ===
            st.subheader("🐝 SHAP Beeswarm: Importancia global del modelo")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            shap.plots.beeswarm(shap_assets['shap_values_global'], show=False)
            st.pyplot(fig1)
            fig1.savefig("shap_beeswarm.png", bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig1)

            # === Waterfall para la muestra actual ===
            explanation = shap.Explanation(
                values=shap_values2,
                base_values=base_values2,
                data=feature_values2,
                feature_names=["Humedad", "Proteína", "Grasa", "Ceniza", "Fibra Cruda",
                            "Carb. Totales", "Fibra Dietética", "Azúcares"]  # Ajusta los nombres
            )
            st.subheader("📊 SHAP Waterfall")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(explanation[0], show=False)
            st.pyplot(fig2)
            fig2.savefig("shap_waterfall.png", bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig2)

            # Generar informe
            pdf_data = generate_report_with_shap(
                data=row_display2,
                frap_value=frap2,
                beeswarm_img="shap_beeswarm.png",
                waterfall_img="shap_waterfall.png",
                recommendations=comment2 if comment2 else None
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



# --- Acerca de ---
with tab5:
    # --- Sección: Información de la Aplicación ---
    with st.expander("📌 Información General", expanded=True):
        st.markdown("""
        **Versión:** 1.0
        **Última actualización:** 06/08/2025
        **Desarrollado por:** [INSERTAR NOMBRES]
        **Repositorio:** [INSERTAR LINK A GITHUB SI ES PÚBLICO]

        Esta aplicación permite predecir la capacidad antioxidante (FRAP) de residuos agroindustriales
        basándose en su composición proximal, utilizando modelos de Machine Learning.
        """)

    # --- Sección: Disclaimer y Limitaciones ---
    with st.expander("⚠️ Limitaciones del Modelo", expanded=False):
        st.markdown("""
        ### Consideraciones importantes:

        1. **Datos de entrenamiento:**
           - El modelo fue entrenado con aproximadamente **900 muestras** de residuos agroindustriales.
           - El coeficiente de determinación (R²) alcanzado fue de **0.68**.

        2. **Alcance predictivo:**
           - El modelo utiliza **regresión Ridge** para las predicciones.
           - Las predicciones son más confiables dentro del rango de valores observados en los datos de entrenamiento.
           - Los valores extremos (fuera de los percentiles 5-95% de los datos de entrenamiento) pueden ser menos precisos.

        3. **Interpretación de resultados:**
           - Los valores FRAP predichos deben considerarse como **estimaciones preliminares**.
           - Se recomienda **validación experimental** para aplicaciones críticas.
           - Las recomendaciones generadas son sugerencias basadas en patrones estadísticos.
        """)

    # --- Sección: Instrucciones de Uso ---
    with st.expander("📚 Guía de Uso", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🔍 Pestañas principales:

            1. **Predicción Individual**
               - Ingresa valores manualmente de composición proximal
               - Obtén predicción FRAP y explicación SHAP
               - Genera informe descargable

            2. **Predicción por Lotes**
               - Sube un archivo CSV con múltiples muestras
               - Descarga resultados y análisis comparativo
               - Genera informe completo con SHAP para todas las muestras

            3. **Simulador What-if**
               - Explora cómo cambios en componentes afectan el FRAP
               - Compara escenarios alternativos

            3. **Búsqueda con IA**
               - Ingresa el nombre de un residuo agroindustrial
               - Obtén por medio de IA (GPT-4 OpenAI) la composición proximal estimada del residuo
               - Obtén predicción FRAP y explicación SHAP
               - Genera informe descargable
            """)

        with col2:
            st.markdown("""
            ### 📊 Formato de datos:

            - **CSV para predicción por lotes** debe contener columnas con:
              `sample_name, moisture, protein, fat, ash, crude_fiber, total_carbohydrates, dietary_fiber, sugars`
              (Descarga plantilla en el sidebar)

            ### 🧪 Unidades de medida:

            - Todos los componentes en **% peso seco** (g/100g)
            - FRAP en **mmol Fe²⁺/100g** de muestra seca

            ### 🔄 Recomendaciones:

            - Para mejores resultados, mantén los valores dentro de rangos razonables
            - Verifica que la suma de componentes no exceda 100%
            """)

    # --- Sección: Marco Teórico ---
    with st.expander("🔬 Fundamentos Científicos", expanded=False):
        st.markdown("""
        ### Método FRAP (Ferric Reducing Antioxidant Power)

        - **Principio:** Mide la capacidad de una muestra para reducir el ion férrico (Fe³⁺) a ferroso (Fe²⁺)
        - **Ventajas:** Simple, reproducible, ampliamente usado en estudios de alimentos
        - **Limitaciones:** Solo detecta antioxidantes reductores en condiciones ácidas

        ### Modelo de Machine Learning

        - **Algoritmo:** Regresión Ridge (L2 regularization)
        - **Variables predictoras:** 8 componentes proximales
        - **Validación:** Cross-validation (5 folds)
        - **Métricas:**
          - R²:
          - RMSE: -- mmol Fe²⁺/100g

        ### Interpretación SHAP

        - Explica cómo cada variable contribuye a la predicción
        - Valores positivos aumentan el FRAP predicho
        - Valores negativos lo disminuyen
        """)

    # --- Sección: Tecnologías utilizadas ---
    with st.expander("Tecnologías utilizadas", expanded=False):
        st.markdown("""

        ### Tecnologías utilizadas:
        - Python, Scikit-learn, SHAP, Streamlit
        - Desplegado en [INSERTAR PLATAFORMA DE DEPLOY]

        """)
        st.markdown("""
<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
    <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
    <img src="https://img.shields.io/badge/SHAP-0.42%2B-FF6D01?logo=shap&logoColor=white" alt="SHAP">
    <img src="https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white" alt="Pandas">
    <img src="https://img.shields.io/badge/Matplotlib-3.7%2B-11557C?logo=matplotlib&logoColor=white" alt="Matplotlib">
    <img src="https://img.shields.io/badge/Numpy-1.24%2B-013243?logo=numpy&logoColor=white" alt="NumPy">
    <img src="https://img.shields.io/badge/Plotly-5.15%2B-3F4F75?logo=plotly&logoColor=white" alt="Plotly">
</div>
""", unsafe_allow_html=True)

    # --- Footer ---
    st.markdown("---")
    st.caption("""
    *Esta aplicación es para fines de investigación. Los resultados no constituyen asesoramiento profesional.*
    """)


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
    - **Alto**: > 40 mmol Fe2+/100g
    - **Medio**: 15-40 mmol Fe2+/100g
    - **Bajo**: < 15 mmol Fe2+/100g
    """)
    try:
        with open("AgroWaste_App/dataset/Ejemplo.csv", "r") as f:
            csv_example = f.read()
        st.sidebar.download_button("📥 Ejemplo CSV", csv_example, "ejemplo_datos.csv", "text/csv")
    except: pass
