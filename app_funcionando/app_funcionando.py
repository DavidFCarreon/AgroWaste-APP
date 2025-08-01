# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime
import os
from jinja2 import Template

try:
    from modelo_frap import predict_frap, get_shap_explainer, get_expected_value, get_clean_feature_names
except ImportError as e:
    st.error("❌ No se pudo cargar modelo_frap.py"); st.stop()

# Configuración de la página
st.set_page_config(
    page_title="FRAP Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("🌱 AgroWaste-APP: Antioxidant Power Predictor")
st.markdown("""
**Aplicación para predecir actividad antioxidante (FRAP) a partir de la composición proximal de residuos agroindustriales**
""")

# Pestañas principales
tab1, tab2, tab3 = st.tabs(["Predicción Individual", "Predicción por Lotes", "Simulador"])


# --- Función: Generar informe con SHAP ---
def generate_report_with_shap(data, frap_value, beeswarm_img, waterfall_img):
    import os
    import base64
    from datetime import datetime

    classification = "Alto" if frap_value > 50 else "Medio" if frap_value > 20 else "Bajo"
    interpretation = {"Alto": "Alto potencial funcional", "Medio": "Potencial moderado", "Bajo": "Bajo potencial"}[classification]
    recommendation = {"Alto": "Priorizar", "Medio": "Considerar", "Bajo": "Descartar"}[classification]

    try:
        def img_to_base64(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        beeswarm_b64 = img_to_base64(beeswarm_img)
        waterfall_b64 = img_to_base64(waterfall_img)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Informe FRAP - {data['sample_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 30px; }}
                h1, h2 {{ color: #2c3e50; }}
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
                <p><strong>Recomendación:</strong> {recommendation}</p>
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
            <p>Los siguientes gráficos muestran cómo cada componente contribuyó a la predicción.</p>

            <h3>🐝 SHAP Beeswarm</h3>
            <img src="data:image/png;base64,{beeswarm_b64}" alt="SHAP Beeswarm">
            <p class="img-caption">Importancia de features en el conjunto de predicciones.</p>

            <h3>📊 SHAP Waterfall</h3>
            <img src="data:image/png;base64,{waterfall_b64}" alt="SHAP Waterfall">
            <p class="img-caption">Contribución paso a paso hacia la predicción final.</p>

            <p><strong>Valor base (FRAP promedio):</strong> {get_expected_value():.2f}</p>
            <p><strong>Predicción final:</strong> {frap_value:.2f}</p>
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





required_cols = ['moisture','protein','fat','ash','crude_fiber','total_carbohydrates','dietary_fiber','sugars']

# --- tab1: Predicción Individual ---
with tab1:
    st.header("Predicción Individual")
    with st.expander("Ingresar datos manualmente"):
        col1, col2 = st.columns(2)
        sample_name = col1.text_input("Nombre", "Muestra 1")
        origin = col2.text_input("Origen", "Agroresiduo")
        cols = st.columns(4)
        moisture = cols[0].number_input("Humedad", 0.0, 95.0, 10.0)
        protein = cols[1].number_input("Proteína", 0.0, 50.0, 15.0)
        fat = cols[2].number_input("Grasa", 0.0, 50.0, 5.0)
        ash = cols[3].number_input("Cenizas", 0.0, 20.0, 4.0)
        cols2 = st.columns(4)
        crude_fiber = cols2[0].number_input("Fibra cruda", 0.0, 80.0, 20.0)
        dietary_fiber = cols2[1].number_input("Fibra dietética", 0.0, 80.0, 22.0)
        total_carbohydrates = cols2[2].number_input("Carbohid. totales", 0.0, 90.0, 40.0)
        sugars = cols2[3].number_input("Azúcares", 0.0, 50.0, 5.0)

        if st.button("Predecir FRAP"):
            row = {
                'moisture': moisture,
                'protein': protein,
                'fat': fat,
                'ash': ash,
                'crude_fiber': crude_fiber,
                'total_carbohydrates': total_carbohydrates,
                'dietary_fiber': dietary_fiber,
                'sugars': sugars
            }
            row_display = row.copy()
            row_display['sample_name'] = sample_name
            row_display['origin'] = origin

            try:
                frap = predict_frap(row)
                st.success(f"FRAP: **{frap:.2f} mmol Fe2+/100g**")

                # === Generar SHAP values ===
                input_df = pd.DataFrame([row])
                explainer = get_shap_explainer()
                shap_values = explainer(input_df)
                shap_values[0].feature_names = get_clean_feature_names()

                # === Beeswarm ===
                st.subheader("🐝 SHAP Beeswarm")
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(fig1)
                fig1.savefig("shap_beeswarm.png", bbox_inches='tight', dpi=150, facecolor='white')
                plt.close(fig1)

                # === Waterfall ===
                st.subheader("📊 SHAP Waterfall")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig2)
                fig2.savefig("shap_waterfall.png", bbox_inches='tight', dpi=150, facecolor='white')
                plt.close(fig2)

                st.write(f"**Valor base (FRAP promedio):** {get_expected_value():.2f}")
                st.write(f"**Predicción final:** {frap:.2f}")

                # === Generar informe ===
                pdf_data = generate_report_with_shap(
                    data=row_display,
                    frap_value=frap,
                    beeswarm_img="shap_beeswarm.png",
                    waterfall_img="shap_waterfall.png"
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




# --- Análisis por lote ---
with tab2:
    # --- Carga de datos ---
    st.header("Carga de datos")
    uploaded_file = st.file_uploader("Sube CSV", type="csv")
    df = pd.read_csv(uploaded_file) if uploaded_file else None

    try:
        feature_names_clean = get_clean_feature_names()
    except:
        feature_names_clean = [
            "Moisture", "Protein", "Fat", "Ash",
            "Crude Fiber", "Total Carbohydrates",
            "Dietary Fiber", "Sugars"
        ]
    if df is not None:
        st.header("3. Análisis por lote")
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Faltan: {missing}")
        else:
            df['FRAP_predicho'] = df.apply(predict_frap, axis=1)
            df['Clasificación'] = df['FRAP_predicho'].apply(lambda x: "Alto" if x > 50 else "Medio" if x > 20 else "Bajo")
            st.dataframe(df.sort_values("FRAP_predicho", ascending=False))

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=df, x='sample_name', y='FRAP_predicho', hue='Clasificación', dodge=False, ax=ax)
            plt.xticks(rotation=45); ax.set_title("FRAP predicho"); st.pyplot(fig)

            @st.cache_data
            def to_csv(x): return x.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Resultados CSV", to_csv(df), "resultados.csv", "text/csv")

            # === SHAP para lote ===
            st.subheader("🐝 SHAP Beeswarm (Análisis por lote)")
            input_batch = df[required_cols]
            explainer = get_shap_explainer()
            try:
                shap_values_batch = explainer(input_batch)
                shap_values_batch.feature_names = get_clean_feature_names()

                fig3, ax3 = plt.subplots(figsize=(8, 6))
                shap.plots.beeswarm(shap_values_batch, show=False)
                st.pyplot(fig3)
                fig3.savefig("shap_beeswarm_lote.png", bbox_inches='tight', dpi=150, facecolor='white')
                plt.close(fig3)
            except Exception as e:
                st.warning(f"SHAP por lote no disponible: {e}")


# --- Simulador "What-if" mejorado ---
with tab3:
    st.header("4. Simulador")
    with st.expander("What-if"):

        # Crear lista de opciones: muestra manual + muestras por lote
        opciones = []
        datos_opciones = []

        # Añadir muestra manual si fue ingresada
        if 'moisture' in locals():
            muestra_manual = {
                'sample_name': sample_name or "Muestra Manual",
                'moisture': moisture,
                'protein': protein,
                'fat': fat,
                'ash': ash,
                'crude_fiber': crude_fiber,
                'total_carbohydrates': total_carbohydrates,
                'dietary_fiber': dietary_fiber,
                'sugars': sugars
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

            try:
                frap_orig = predict_frap(base_row)
                frap_mod = predict_frap(modified)

                st.metric("Original", f"{frap_orig:.2f}")
                st.metric("Modificado", f"{frap_mod:.2f}", delta=f"{frap_mod - frap_orig:+.2f}")

                if frap_mod > frap_orig:
                    st.success("✅ El cambio aumentaría el potencial antioxidante.")
                elif frap_mod < frap_orig:
                    st.warning("⚠️ El cambio reduciría el potencial.")
                else:
                    st.info("➡️ Sin cambio significativo.")
            except Exception as e:
                st.error(f"Error en simulación: {e}")

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
