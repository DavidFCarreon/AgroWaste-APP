# crear_proyecto_completo.py
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import Ridge
import joblib

print("🚀 Creando proyecto AgroWaste-APP...")

# ===================================
# 1. Crear carpetas
# ===================================
folders = [
    "models",
    "data",
    "templates"
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
print("✅ Carpetas creadas")

# ===================================
# 2. Generar dataset simulado (si no existe)
# ===================================
data_file = "data_preprocessed_FRAP_final.csv"
if not os.path.exists(data_file):
    np.random.seed(42)
    n = 300

    # Simular composición proximal realista
    data_sim = pd.DataFrame({
        'moisture': np.random.uniform(5, 95, n),
        'protein': np.random.uniform(2, 30, n),
        'fat': np.random.uniform(0.5, 25, n),
        'ash': np.random.uniform(0.5, 10, n),
        'crude_fiber': np.random.uniform(5, 40, n),
        'total_carbohydrates': np.random.uniform(20, 80, n),
        'dietary_fiber': np.random.uniform(10, 35, n),
        'sugars': np.random.uniform(1, 20, n)
    })

    # FRAP basado en correlaciones observadas + ruido
    data_sim['FRAP'] = (
        0.78 * data_sim['protein'] +
        0.65 * data_sim['ash'] +
        0.65 * data_sim['crude_fiber'] +
        0.68 * data_sim['dietary_fiber'] +
        0.56 * data_sim['total_carbohydrates'] +
        0.39 * data_sim['fat'] -
        0.73 * data_sim['moisture'] -
        0.16 * data_sim['sugars']
    )
    data_sim['FRAP'] = np.clip(2.0 * data_sim['FRAP'] + np.random.normal(0, 2, n), 5, 80)

    data_sim.to_csv(data_file, index=False)
    print("✅ Dataset simulado guardado como data_preprocessed_FRAP_final.csv")
else:
    print("✅ Usando dataset existente")

# ===================================
# 3. entrenar_y_guardar_modelo.py
# ===================================
train_script = '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import Ridge
import joblib
import os

print("📂 Cargando datos...")
df = pd.read_csv("data_preprocessed_FRAP_final.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

numerical_features = selector(dtype_exclude="object")(X_train)
categorical_features = selector(dtype_include="object")(X_train)

numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="passthrough")

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge())
])

param_dist = {"regressor__alpha": np.logspace(-2, 2, 100)}
random_search = RandomizedSearchCV(
    model_pipeline, param_distributions=param_dist, n_iter=50,
    cv=5, scoring="neg_mean_squared_error", random_state=42, n_jobs=-1
)

print("🔍 Entrenando modelo...")
random_search.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(random_search.best_estimator_, "models/modelo_frap_real.pkl")
print("✅ Modelo guardado en models/modelo_frap_real.pkl")
'''.strip()

with open("entrenar_y_guardar_modelo.py", "w", encoding="utf-8") as f:
    f.write(train_script)
print("✅ entrenar_y_guardar_modelo.py creado")

# ===================================
# 4. Ejecutar entrenamiento automáticamente
# ===================================
print("⚙️ Entrenando modelo (esto puede tardar unos segundos)...")
exec(open("entrenar_y_guardar_modelo.py").read())

# ===================================
# 5. modelo_frap.py
# ===================================
modelo_code = '''
import joblib
import pandas as pd
import numpy as np
import shap
import os

MODEL_PATH = "models/modelo_frap_real.pkl"
DATA_PATH = "data_preprocessed_FRAP_final.csv"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH}")

model = joblib.load(MODEL_PATH)
preprocessor = model.named_steps['preprocessor']

df_full = pd.read_csv(DATA_PATH)
X_background = df_full.iloc[:, :-1]
X_background_transformed = preprocessor.transform(X_background)
X_sampled = shap.sample(X_background_transformed, 100)

explainer = shap.Explainer(
    model.predict,
    X_sampled,
    feature_names=preprocessor.get_feature_names_out()
)

expected_value = float(explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0])
feature_names = preprocessor.get_feature_names_out()
clean_feature_names = [name.split("__")[-1].replace("_", " ").title() for name in feature_names]

def predict_frap(row):
    input_df = pd.DataFrame([row])
    pred = model.predict(input_df)[0]
    return float(np.round(pred, 2))

def get_shap_explainer():
    return explainer

def get_expected_value():
    return expected_value

def get_clean_feature_names():
    return clean_feature_names
'''.strip()

with open("modelo_frap.py", "w", encoding="utf-8") as f:
    f.write(modelo_code)
print("✅ modelo_frap.py creado")

# ===================================
# 6. ejemplo_datos.csv
# ===================================
ejemplo = pd.DataFrame({
    'sample_name': ['Bagazo de caña', 'Cáscara de naranja', 'Orujo de oliva'],
    'origin': ['Sacarosa', 'Cítricos', 'Olivas'],
    'moisture': [10.2, 8.5, 6.8],
    'protein': [3.1, 6.2, 12.4],
    'fat': [1.8, 0.5, 5.1],
    'ash': [7.2, 3.8, 2.9],
    'crude_fiber': [28.5, 18.0, 25.3],
    'total_carbohydrates': [45.0, 60.0, 45.0],
    'dietary_fiber': [32.0, 22.5, 30.1],
    'sugars': [2.1, 5.2, 1.8]
})
ejemplo.to_csv("data/ejemplo_datos.csv", index=False)
print("✅ ejemplo_datos.csv guardado")

# ===================================
# 7. templates/informe.html
# ===================================
html_template = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Informe FRAP - {{sample_name}}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; }
        h1, h2 { color: #2c3e50; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .highlight { background-color: #e8f5e8; padding: 10px; border-left: 4px solid green; }
    </style>
</head>
<body>
    <h1>Informe de Evaluación de Potencial Antioxidante</h1>
    <p><strong>Muestra:</strong> {{sample_name}}</p>
    <p><strong>Origen:</strong> {{origin}}</p>
    <p><strong>Fecha:</strong> {{date}}</p>

    <h2>Composición Proximal</h2>
    <table>
        <tr><th>Componente</th><th>Valor (%)</th></tr>
        <tr><td>Humedad</td><td>{{moisture}}</td></tr>
        <tr><td>Proteína</td><td>{{protein}}</td></tr>
        <tr><td>Grasa</td><td>{{fat}}</td></tr>
        <tr><td>Cenizas</td><td>{{ash}}</td></tr>
        <tr><td>Fibra cruda</td><td>{{crude_fiber}}</td></tr>
        <tr><td>Carbohidratos totales</td><td>{{total_carbohydrates}}</td></tr>
        <tr><td>Fibra dietética</td><td>{{dietary_fiber}}</td></tr>
        <tr><td>Azúcares</td><td>{{sugars}}</td></tr>
    </table>

    <h2>Predicción de Actividad Antioxidante (FRAP)</h2>
    <div class="highlight">
        <p><strong>FRAP predicho:</strong> {{frap_value}} mmol Fe2+/100g</p>
        <p><strong>Clasificación:</strong> {{classification}}</p>
        <p><strong>Interpretación:</strong> {{interpretation}}</p>
    </div>

    <h2>Recomendación para I+D</h2>
    <p>{{recommendation}}</p>
</body>
</html>'''.strip()

with open("templates/informe.html", "w", encoding="utf-8") as f:
    f.write(html_template)
print("✅ informe.html guardado")

# ===================================
# 8. app.py
# ===================================
app_code = '''
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

st.set_page_config(page_title="AgroWaste-APP", layout="wide")
st.title("🔬 AgroWaste-APP: Antioxidant Power Predictor")

def generate_report(data, frap_value):
    classification = "Alto" if frap_value > 50 else "Medio" if frap_value > 20 else "Bajo"
    interpretation = {"Alto": "Alto potencial funcional", "Medio": "Potencial moderado", "Bajo": "Bajo potencial"}[classification]
    recommendation = {"Alto": "Priorizar", "Medio": "Considerar", "Bajo": "Descartar"}[classification]

    try:
        with open("templates/informe.html", "r", encoding="utf-8") as f:
            template = Template(f.read())
        html_out = template.render(
            sample_name=data["sample_name"],
            origin=data.get("origin", "N/A"),
            moisture=data["moisture"],
            protein=data["protein"],
            fat=data["fat"],
            ash=data["ash"],
            crude_fiber=data["crude_fiber"],
            total_carbohydrates=data["total_carbohydrates"],
            dietary_fiber=data["dietary_fiber"],
            sugars=data["sugars"],
            frap_value=frap_value,
            classification=classification,
            interpretation=interpretation,
            recommendation=recommendation,
            date=datetime.now().strftime("%d/%m/%Y")
        )
        with open("informe_temp.html", "w", encoding="utf-8") as f:
            f.write(html_out)
        from weasyprint import HTML
        HTML("informe_temp.html").write_pdf("informe.pdf")
        with open("informe.pdf", "rb") as f:
            pdf_data = f.read()
        os.remove("informe_temp.html")
        return pdf_data
    except ImportError:
        st.warning("💡 Para PDF: pip install weasyprint"); return None
    except Exception as e:
        st.error(f"❌ Error PDF: {e}"); return None

st.sidebar.info("1. Sube CSV o ingresa muestra\\n2. Predice FRAP\\n3. Descarga informe")
try:
    with open("data/ejemplo_datos.csv", "r") as f:
        csv_example = f.read()
    st.sidebar.download_button("📥 Ejemplo CSV", csv_example, "ejemplo_datos.csv", "text/csv")
except: pass

st.header("1. Carga de datos")
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

required_cols = ['moisture','protein','fat','ash','crude_fiber','total_carbohydrates','dietary_fiber','sugars']

st.header("2. Muestra manual")
with st.expander("Ingresar"):
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
            'moisture': moisture, 'protein': protein, 'fat': fat, 'ash': ash,
            'crude_fiber': crude_fiber, 'total_carbohydrates': total_carbohydrates,
            'dietary_fiber': dietary_fiber, 'sugars': sugars
        }
        row_display = row.copy()
        row_display['sample_name'] = sample_name
        row_display['origin'] = origin

        try:
            frap = predict_frap(row)
            st.success(f"FRAP: **{frap:.2f} mmol Fe2+/100g**")
            pdf_data = generate_report(row_display, frap)
            if pdf_data:
                st.download_button("📥 Descargar informe PDF", pdf_data, f"informe_{sample_name}.pdf", "application/pdf")

            # SHAP Beeswarm
            st.subheader("🐝 SHAP Beeswarm")
            input_df = pd.DataFrame([row])
            explainer = get_shap_explainer()
            shap_values = explainer(input_df)
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig1)
            plt.close(fig1)

            # SHAP Waterfall
            st.subheader("📊 SHAP Waterfall")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig2)
            plt.close(fig2)

            st.write(f"**Valor base (FRAP promedio):** {get_expected_value():.2f}")
            st.write(f"**Predicción final:** {frap:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")

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
        if st.button("Generar informe PDF por lote"):
            df_html = df.to_html(index=False)
            html = f"""<!DOCTYPE html><html><body>
            <h1>Informe por Lote - FRAP Predicted</h1>{df_html}</body></html>"""
            with open("lote_temp.html", "w") as f: f.write(html)
            from weasyprint import HTML
            HTML("lote_temp.html").write_pdf("informe_lote.pdf")
            with open("informe_lote.pdf", "rb") as f: pdf_data = f.read()
            os.remove("lote_temp.html")
            st.download_button("📥 Descargar informe por lote", pdf_data, "informe_lote.pdf", "application/pdf")

st.header("4. Simulador")
with st.expander("What-if"):
    if df is not None and len(df) > 0:
        base_sample = st.selectbox("Muestra", df['sample_name'])
        base_row = df[df['sample_name'] == base_sample].iloc[0]
        component = st.selectbox("Modificar", required_cols)
        change = st.slider(f"Δ {component}", -20.0, 20.0, 0.0, 0.5)
        modified = base_row.copy(); modified[component] += change; modified[component] = max(0, modified[component])
        frap_orig = predict_frap(base_row)
        frap_mod = predict_frap(modified)
        st.metric("Original", f"{frap_orig:.2f}")
        st.metric("Modificado", f"{frap_mod:.2f}", delta=f"{frap_mod - frap_orig:+.2f}")
        if frap_mod > frap_orig: st.success("✅ Aumenta")
        elif frap_mod < frap_orig: st.warning("⚠️ Disminuye")
        else: st.info("➡️ Igual")
'''.strip()

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)
print("✅ app.py creado")

# ===================================
# 9. Finalizar
# ===================================
print("\n🎉 ¡Proyecto completado!")
print("📌 Ejecuta: streamlit run app.py")
print("💡 Asegúrate de tener instalado: pip install streamlit pandas numpy scikit-learn shap matplotlib seaborn jinja2 weasyprint joblib")
