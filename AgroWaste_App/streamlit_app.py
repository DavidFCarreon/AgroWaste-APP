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
from app_utils import get_clean_feature_names, safe_api_request, process_batch_shap, generate_report_with_shap , generate_batch_report_with_shap



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
        response = safe_api_request(url_pred, params)
        prediction = response.json()
        frap = prediction.get("FRAP_value", 0)
        return float(np.round(frap, 2))  # Aseguramos que sea un float redondeado a 2 decimales

    except Exception as e:
        st.warning(f"Excepción para muestra {row.get('sample_name', '')}: {str(e)}")
        return None

# Cargamos los assets al inicio
shap_assets = load_shap_assets()

try:
    # Modificación: Ahora get_clean_feature_names y get_background_data vienen de los assets
    get_clean_feature_names = lambda: shap_assets['feature_names']
    get_background_data = lambda: shap_assets['background_df']

except ImportError as e:
    st.error("❌ No se pudo cargar modelo_frap.py"); st.stop()

# --- Configuración original de la página ---
st.set_page_config(
    page_title="AgroWaste-APP",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'samples_db' not in st.session_state:
    st.session_state.samples_db = []


# --- Título y descripción ---
st.title("🌱 AgroWaste-APP: Antioxidant Power Predictor")
st.markdown("""
**Aplicación para predecir actividad antioxidante (FRAP) a partir de la composición proximal de residuos agroindustriales**
""")

# --- Pestañas principales ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Predicción Individual", "Predicción por Lotes", "Búsqueda con IA", "Simulador", "Acerca de"])

required_cols = ['moisture','protein','fat','ash','crude_fiber','total_carbohydrates','dietary_fiber','sugars']

# --- tab1: Predicción Individual (optimizada) ---
with tab1:
    st.header("Predicción Individual")
    col1, col2 = st.columns(2)
    sample_name = col1.text_input("Nombre del residuo agroindustrial. Ingrese nombre para obtener recomendaciones personalizadas", "Orujo de oliva")
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
        row_display['sample_name'] = sample_name
        row_display['origin'] = origin

        params1 = {
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
            response1 = safe_api_request(url_pred, params1)
            prediction1 = response1.json()
            frap1 = prediction1.get("FRAP_value", 0)
            shap_values1 = np.array(prediction1["shap_values"])
            base_values1 = np.array(prediction1["shap_base_values"])
            feature_values1 = np.array(prediction1["shap_data"])
            # Mostrar el resultado con estilo condicional
            if frap1 < 15:
                st.warning(f"FRAP: **{frap1:.2f} mmol Fe²⁺/100g** - *Poder antioxidante bajo*")
            elif 15 <= frap1 < 40:
                st.info(f"FRAP: **{frap1:.2f} mmol Fe²⁺/100g** - *Poder antioxidante medio*")
            else:  # frap >= 40
                st.success(f"FRAP: **{frap1:.2f} mmol Fe²⁺/100g** - *Poder antioxidante alto*")


            # === Recomendaciones dinámicas ===
            params_gc = {
                'FRAP_value': frap1,
                'product_name': sample_name,
            }
            url_gc = "https://agrowaste-app-476771143854.europe-west1.run.app/get_comments"

            response_gc = safe_api_request(url_gc, params_gc)

            recom = response_gc.json()
            comment = recom.get("Comments", 0)

            # Título con estilo personalizado
            st.markdown("""
            <h3 style='font-size: 24px; color: #2c3e50; margin-bottom: 10px;'>
                🔬 Recomendaciones de I+D para la muestra ingresada:
            </h3>
            """, unsafe_allow_html=True)

            # Contenido con estilo personalizado
            st.markdown(f"""
            <div style='font-size: 20px; line-height: 1.6; background-color: #f8f9fa;
                        padding: 15px; border-radius: 8px; border-left: 4px solid #4e79a7;text-align: justify'>
                {comment}
            </div>
            """, unsafe_allow_html=True)

            # === SHAP ===
            st.subheader("🐝 SHAP Beeswarm: Importancia global del modelo")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            shap.plots.beeswarm(shap_assets['shap_values_global'], show=False)
            st.pyplot(fig1)
            fig1.savefig("shap_beeswarm.png", bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig1)

            # === Waterfall para la muestra actual (se calcula solo esto) ===
            explanation = shap.Explanation(
                values=shap_values1,
                base_values=base_values1,
                data=feature_values1,
                feature_names=["Humedad", "Proteína", "Grasa", "Ceniza", "Fibra Cruda",
                            "Carb. Totales", "Fibra Dietética", "Azúcares"]  # Ajusta los nombres
            )
            st.subheader("📊 SHAP Waterfall")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(explanation[0], show=False)
            st.pyplot(fig2)
            fig2.savefig("shap_waterfall.png", bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig2)

            # --- Guardar muestra en base de datos temporal ---
            sample_to_save = row_display.copy()
            sample_to_save['source'] = "Predicción Individual"
            sample_to_save['id'] = f"ind_{datetime.now().strftime('%H%M%S')}"  # ID único
            # Evitar duplicados por nombre
            st.session_state.samples_db = [
                s for s in st.session_state.samples_db
                if s['sample_name'] != sample_to_save['sample_name'] or s['source'] != "Predicción Individual"
            ]
            st.session_state.samples_db.append(sample_to_save)

            # Generar informe
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



# --- tab2: Predicción por Lotes (optimizada) ---
with tab2:
    # --- Carga de datos ---
    st.header("Carga de datos")
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv", help="El archivo debe contener las columnas requeridas")
    df = pd.read_csv(uploaded_file) if uploaded_file else None

    # Columnas requeridas (nombres originales que espera el modelo)
    #required_cols = ['moisture','protein','fat','ash','crude_fiber',
    #               'total_carbohydrates','dietary_fiber','sugars']

    # Diccionario para traducción de nombres de columnas (sin %)
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

        # Verificar columnas requeridas
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            missing_display = [display_names.get(col, col) for col in missing]
            st.error(f"🚫 Faltan columnas requeridas: {', '.join(missing_display)}")
        else:
            # Crear nombre de muestra si no existe
            if 'sample_name' not in df.columns:
                df['sample_name'] = [f'Muestra {i+1}' for i in range(len(df))]

            # URL de la API
            url_pred = "https://agrowaste-app-476771143854.europe-west1.run.app/predict"

            # Realizar predicciones
            with st.spinner("Realizando predicciones..."):
                df['FRAP_predicho'] = df.apply(predict_row, axis=1)
                df['Clasificación'] = df['FRAP_predicho'].apply(lambda x: "Alto" if x > 40 else "Medio" if x > 15 else "Bajo")

            # Crear DataFrame para visualización
            display_df = df.copy()
            cols_to_rename = {k: v for k, v in display_names.items() if k in display_df.columns}
            display_df = display_df.rename(columns=cols_to_rename)

           # --- Mostrar tabla con estilo mejorado ---
            st.markdown("### Resultados del análisis")

            # Configuración CSS definitiva para centrado perfecto
            st.markdown("""
            <style>
                /* Contenedor principal */
                div[data-testid="stDataFrame"] {
                    width: 100% !important;
                    margin: 0 auto !important;
                }

                /* Cabeceras de columna */
                div[data-testid="stDataFrame"] th {
                    background-color: #f8f9fa !important;
                    color: #2c3e50 !important;
                    font-weight: bold !important;
                    text-align: center !important;
                    padding: 8px !important;
                }

                /* Celdas de datos */
                div[data-testid="stDataFrame"] td {
                    text-align: center !important;
                    vertical-align: middle !important;
                    padding: 8px !important;
                }

                /* Forzar centrado en todos los elementos */
                .stDataFrame .col-header {
                    text-align: center !important;
                    justify-content: center !important;
                }

                .stDataFrame .data {
                    text-align: center !important;
                    justify-content: center !important;
                }

                /* Ajustar el ancho de las columnas */
                .stDataFrame table {
                    table-layout: auto !important;
                    width: 100% !important;
                }

                /* Mejorar el hover */
                div[data-testid="stDataFrame"] tr:hover {
                    background-color: #f1f1f1 !important;
                }
            </style>
            """, unsafe_allow_html=True)

            # Mostrar DataFrame con columnas ajustadas
            st.dataframe(
                display_df.sort_values("FRAP Predicho (mmol Fe²⁺/100g)", ascending=False),
                height=min(600, 35 * len(display_df) + 40),
                use_container_width=True,
                hide_index=True,
                column_config={
                    col: st.column_config.Column(
                        label=col,
                        width="small" if col != "Residuo agroindustrial" else "medium",
                    ) for col in display_df.columns
                }
            )

            # --- Visualizaciones adicionales ---
            st.markdown("---")

            # --- Gráfico de barras premium ---
            import matplotlib.patheffects as path_effects
            from matplotlib import rcParams

            # Configurar estilo moderno
            plt.style.use('default')  # Resetear estilo
            rcParams.update({
                'axes.facecolor': '#f8f9fa',
                'grid.color': '#e1e4e8',
                'axes.edgecolor': '#bdc3c7',
                'axes.labelcolor': "#10181f",
                'xtick.color': '#2c3e50',
                'ytick.color': '#2c3e50',
                'axes.titlepad': 20
            })

            # Paleta de colores mejorada
            palette = {
                'Alto': "#7ADAA5",  # Verde esmeralda más intenso
                'Medio': "#239BA7",  # Azul más profesional
                'Bajo': '#E1AA36'   # Amarillo dorado cálido
            }

            # Configuración de la figura
            fig, ax = plt.subplots(figsize=(13, 7))
            fig.patch.set_facecolor('white')

            # Crear el gráfico con bordes estilizados
            barplot = sns.barplot(
                data=df,
                x='sample_name',
                y='FRAP_predicho',
                hue='Clasificación',
                palette=palette,
                dodge=False,
                ax=ax,
                saturation=0.85,
                linewidth=1.5,
                edgecolor='white',
                alpha=0.93
            )

            # Efecto de esquinas redondeadas (simulado)
            for bar in barplot.patches:
                bar.set_path_effects([
                    path_effects.withStroke(
                        linewidth=2.5,
                        foreground="white",
                        alpha=0.8
                    )
                ])

            # Personalización avanzada
            ax.set_title("Potencial Antioxidante FRAP por Muestra",
                        fontsize=15, pad=20, fontweight='bold', color='#2c3e50')
            ax.set_ylabel(r"FRAP$_{\text{pred}}$ (mmol Fe$^{2+}$/100g)",
                        fontsize=12, labelpad=12)
            ax.set_xlabel("Muestras Analizadas", fontsize=12, labelpad=12)

            # Rotación y estilo de etiquetas
            plt.xticks(rotation=45, ha='right', fontsize=11, rotation_mode='anchor')
            plt.yticks(fontsize=11)

            # Añadir valores con estilo mejorado
            for p in barplot.patches:
                height = p.get_height()
                # Verificar si la altura es mayor a un umbral (por ejemplo, 0.01)
                if height > 0.01:  # Puedes ajustar este umbral según tus necesidades
                    ax.annotate(
                        f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center',
                        va='center',
                        xytext=(0, 7),
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='bold',
                        color="#0f161d",
                        bbox=dict(
                            boxstyle='round,pad=0.2',
                            facecolor='white',
                            edgecolor='none',
                            alpha=0.7
                        )
                    )

            # Leyenda premium
            legend = ax.legend(
                title='Clasificación FRAP',
                title_fontsize=12,
                fontsize=11,
                frameon=True,
                framealpha=0.95,
                facecolor='white',
                edgecolor="#000000",
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderpad=1
            )
            legend.get_frame().set_linewidth(1.5)
            legend.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.2")

            # Borde del área del gráfico
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#000000")
                spine.set_linewidth(1.8)

            # Ajustes finales
            plt.tight_layout(pad=2)
            fig.patch.set_alpha(1.0)

            # Mostrar en Streamlit
            st.pyplot(fig)
            plt.close(fig)

            # Botón para descargar CSV
            @st.cache_data
            def to_csv(x):
                return x.to_csv(index=False).encode('utf-8')

            st.download_button("📥 Resultados CSV",
                               to_csv(df),
                               "resultados.csv",
                               "text/csv")

        # --- Mostrar recomendaciones en la interfaz ---
        if df is not None and 'Clasificación' in df.columns:
            st.markdown("""
            <style>
            .recommendation-container {
                background-color: #f8f9fa;
                color: #2c3e50;
                padding: 25px;
                border-radius: 8px;
                border-left: 4px solid #4e79a7;
                text-align: justify;
                margin-bottom: 20px;
            }
            .recommendation-title {
                font-size: 24px;  /* Título principal a 24px */
                color: #2c3e50;
                margin-bottom: 20px;
                font-weight: 600;
            }
            .recommendation-subtitle {
                color: #2c3e50;
                margin-top: 20px;
                margin-bottom: 10px;
                font-size: 22px;  /* Subtítulos a 22px */
                font-weight: 500;
            }
            .recommendation-text {
                font-size: 20px;  /* Texto general a 20px */
                line-height: 1.6;
            }
            .recommendation-list {
                font-size: 20px;  /* Listas a 20px */
                padding-left: 25px;
                margin-top: 10px;
            }
            .recommendation-list li {
                margin-bottom: 8px;
            }
            </style>
            """, unsafe_allow_html=True)

            high_count = len(df[df['Clasificación'] == 'Alto'])
            med_count = len(df[df['Clasificación'] == 'Medio'])
            low_count = len(df[df['Clasificación'] == 'Bajo'])

            # Contenedor principal
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
                    <p><strong>Líneas recomendadas:</strong></p>
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
                    <p><strong>Líneas recomendadas:</strong></p>
                    <ul class="recommendation-list">
                        <li>Desarrollo de extractos con funcionalidad moderada</li>
                        <li>Incorporación como ingrediente funcional complementario</li>
                        <li>Evaluación como fuente de fibra dietética u otros metabolitos</li>
                        <li>Uso como sustrato en procesos biotecnológicos (fermentación, producción de enzimas o metabolitos de valor)</li>
                        <li>Aplicación en formulación de productos combinados con otros residuos que permitan sinergias funcionales</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            if low_count > 0:
                st.markdown(f"""
                <div class="recommendation-text">
                    <h4 class="recommendation-subtitle">Residuos con Baja Capacidad Antioxidante: {low_count}</h4>
                    <p><strong>Estrategia sugerida:</strong> Desviar el enfoque hacia otras fracciones.</p>
                    <p><strong>Líneas recomendadas:</strong></p>
                    <ul class="recommendation-list">
                        <li>Aprovechamiento como fuente de fibra estructural</li>
                        <li>Producción de biocombustibles o bioenergía</li>
                        <li>Uso en alimentación animal o compostaje</li>
                        <li>Aplicaciones en fermentación de estado sólido o líquida para obtención de subproductos industriales (enzimas, ácidos
orgánicos, biopigmentos)</li>
                        <li>Considerar su inclusión como componente de mezclas multirresiduo, en esquemas de valorización combinada</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Cierre del contenedor principal
            st.markdown("</div>", unsafe_allow_html=True)

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

# --- Integración con OpenAI ---
with tab3:
    st.header("Búsqueda con IA (Open AI GPT-4 mini)")
    search_gpt = st.text_input("Ingresa el nombre de un residuo agroindustrial:", "Bagazo de manzana")
    if st.button("Predecir FRAP con IA"):

        # Llamar a la API para obtener composición proximal
        url_search = "https://agrowaste-app-476771143854.europe-west1.run.app/get_features"
        try:
            s_response = safe_api_request(url_search, params={'product_name': search_gpt})

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

            # --- Guardar muestra de IA en base de datos temporal ---
            sample_to_save_ia = row_display2.copy()
            sample_to_save_ia['source'] = "Búsqueda con IA"
            sample_to_save_ia['id'] = f"ia_{datetime.now().strftime('%H%M%S')}"
            # Evitar duplicados por nombre
            st.session_state.samples_db = [
                s for s in st.session_state.samples_db
                if s['sample_name'] != sample_to_save_ia['sample_name'] or s['source'] != "Búsqueda con IA"
            ]
            st.session_state.samples_db.append(sample_to_save_ia)

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

            # === Recomendaciones dinámicas ===
            params_gc2 = {
                'FRAP_value': frap2,
                'product_name': search_gpt,
            }
            url_gc2 = "https://agrowaste-app-476771143854.europe-west1.run.app/get_comments"

            response_gc2 = safe_api_request(url_gc2, params_gc2)

            recom2 = response_gc2.json()
            comment2 = recom2.get("Comments", 0)

            # --- Mismo estilo que tab1 ---
            st.markdown("""
                <h3 style='font-size: 24px; color: #2c3e50; margin-bottom: 10px;'>🔬 Recomendaciones de I+D para la muestra ingresada:</h3>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style='
                    font-size: 18px;
                    line-height: 1.7;
                    background-color: #f8f9fa;
                    padding: 18px;
                    border-radius: 8px;
                    border-left: 4px solid #4e79a7;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: justify;
                    margin-bottom: 20px;
                '>
                    {comment2}
                </div>
                """, unsafe_allow_html=True)


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

# --- Simulador "What-if" ---
with tab4:
    st.header("Simulador")
    st.markdown("""
                Explora cómo cambios en la composición proximal afectan el potencial antioxidante (FRAP) de residuos agroindustriales.
                """)
    with st.expander("What-if"):
        # Diccionario de traducción
        traduccion_componentes = {
            'moisture': 'Humedad',
            'protein': 'Proteína',
            'fat': 'Grasa',
            'ash': 'Ceniza',
            'crude_fiber': 'Fibra Cruda',
            'total_carbohydrates': 'Carb. Totales',
            'dietary_fiber': 'Fibra Diet.',
            'sugars': 'Azúcares'
        }

        # Crear lista de opciones: muestra manual + muestras por lote
        # --- Construir lista de opciones desde todas las fuentes ---
        opciones = []
        datos_opciones = []

        # 1. Muestra del lote (si existe)
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                name = row['sample_name']
                if name not in opciones:
                    opciones.append(name)
                    datos_opciones.append(row.to_dict())

        # 2. Muestras guardadas en session_state (tab1 y tab4)
        for s in st.session_state.samples_db:
            name = s['sample_name']
            if name not in opciones:
                opciones.append(name)
                datos_opciones.append(s.copy())  # copia completa

        # 3. Muestra manual actual (si no está ya en la lista)
        try:
            current_manual_name = sample_name or "Muestra Manual"
            if current_manual_name not in opciones:
                muestra_manual = {
                    'sample_name': current_manual_name,
                    'moisture': Moisture,
                    'protein': Protein,
                    'fat': Fat,
                    'ash': Ash,
                    'crude_fiber': Crude_Fiber,
                    'total_carbohydrates': Total_Carbohydrates,
                    'dietary_fiber': Dietary_Fiber,
                    'sugars': Sugars
                }
                opciones.append(current_manual_name)
                datos_opciones.append(muestra_manual)
        except:
            pass  # Si no están definidas, ignora


        if len(opciones) == 0:
            st.info("No hay muestras disponibles para simular.")
        else:
            base_sample_name = st.selectbox("Selecciona muestra", opciones)
            idx = opciones.index(base_sample_name)
            base_row = datos_opciones[idx]

            # Mostrar los nombres en español pero mantener los valores en inglés internamente
            componente_seleccionado = st.selectbox(
                "Modificar componente",
                options=required_cols,
                format_func=lambda x: traduccion_componentes.get(x, x)
            )

            change = st.slider(
                f"Δ {traduccion_componentes.get(componente_seleccionado, componente_seleccionado)} (%)",
                -20.0, 20.0, 0.0, 0.5
            )

            modified = base_row.copy()
            modified[componente_seleccionado] += change
            modified[componente_seleccionado] = max(0.0, modified[componente_seleccionado])

            # Función para obtener predicción desde API
            def get_frap_prediction(data):
                params_sim = {
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
                    url_pred = "https://agrowaste-app-476771143854.europe-west1.run.app/predict"
                    response_sim = safe_api_request(url_pred, params_sim)
                    frap_sim = response_sim.json().get("FRAP_value", 0)
                    return frap_sim
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





# --- Acerca de ---
with tab5:
    # --- Sección: Información de la Aplicación ---
    with st.expander("📌 Información General", expanded=True):
        st.markdown("""
        - **Versión:** 1.0\n
        - **Última actualización:** 06/08/2025\n
        - **Desarrollado por:**\n
            Itzel Yoali Hernández Montesinos, Nancy Rojas Salvatierra, Julio Pardini Susacasa, David Fernando Carreón Delgado
        - **Repositorio:** https://github.com/DavidFCarreon/AgroWaste-APP

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
               - Obtén por medio de IA (OpenAI GPT-4 mini) la composición proximal estimada del residuo
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
            - Considera validación experimental de los resultados
            - Usa el simulador para explorar diferentes escenarios de composición
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
          - R²: 0.68
          - MSE: 41.09
          - MAE: 5.09
          - MAPE: 0.93

        ### Interpretación SHAP

        - Explica cómo cada variable contribuye a la predicción
        - Valores positivos aumentan el FRAP predicho
        - Valores negativos lo disminuyen
        """)

    # --- Sección: Tecnologías utilizadas ---
    with st.expander("Tecnologías utilizadas", expanded=False):
        st.markdown("""

        ### Tecnologías utilizadas:
        - Python, Scikit-learn, Google Cloud, OpenAI GPT-4 mini, FastAPI, Uvicorn, Streamlit
        - Desplegado en Streamlit Cloud
        """)
        st.markdown("""
<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Scikit--learn-1.0%2B-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
    <img src="https://img.shields.io/badge/SHAP-0.40%2B-FF6D01?logo=shap&logoColor=white" alt="SHAP">
    <img src="https://img.shields.io/badge/Streamlit-1.12%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/Pandas-1.3%2B-150458?logo=pandas&logoColor=white" alt="Pandas">
    <img src="https://img.shields.io/badge/Matplotlib-3.5%2B-11557C?logo=matplotlib&logoColor=white" alt="Matplotlib">
    <img src="https://img.shields.io/badge/Numpy-1.21%2B-013243?logo=numpy&logoColor=white" alt="NumPy">
    <img src="https://img.shields.io/badge/Plotly-5.0%2B-3F4F75?logo=plotly&logoColor=white" alt="Plotly">
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

