# modelo_frap.py
import joblib
import pandas as pd
import numpy as np
import shap
import os

# Rutas
MODEL_PATH = "models/modelo_frap_real.pkl"
DATA_PATH = "data_preprocessed_FRAP_final.csv"

# Validar archivos
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH}")

# Cargar modelo
model = joblib.load(MODEL_PATH)
preprocessor = model.named_steps['preprocessor']

# Cargar datos para background
df_full = pd.read_csv(DATA_PATH)
X_background = df_full.iloc[:, :-1]  # Sin la columna FRAP (target)

# Transformar con el preprocessor del modelo
X_background_transformed = preprocessor.transform(X_background)

# Muestra 100 filas representativas
X_sampled_transformed = shap.sample(X_background_transformed, 100)  # Es un numpy array

# === 🔴 Corrección clave: convertir a DataFrame con nombres de columnas ===
feature_names = X_background.columns.tolist()  # Nombres originales de las columnas
X_sampled_df = pd.DataFrame(X_sampled_transformed, columns=feature_names)

# Crear explainer: ahora usamos el DataFrame original (X_background) para el background
explainer = shap.Explainer(
    model.predict,
    X_sampled_df,
    feature_names=feature_names
)

# === Calcular expected_value de forma segura ===
try:
    if hasattr(explainer, 'expected_value'):
        expected_value = explainer.expected_value
    elif len(explainer.base_values) > 0:
        expected_value = explainer.base_values[0]
    else:
        expected_value = float(model.predict(X_sampled_df).mean())
except:
    expected_value = float(model.predict(X_sampled_df).mean())

# Asegurar que sea un float
if isinstance(expected_value, np.ndarray):
    expected_value = float(expected_value[0])
else:
    expected_value = float(expected_value)

# Limpiar nombres para mostrar en gráficos
clean_feature_names = [name.replace("_", " ").title() for name in feature_names]

def predict_frap(row):
    """
    Predice FRAP a partir de un dict con las features.
    row: dict con keys igual a las columnas de entrenamiento.
    """
    input_df = pd.DataFrame([row])
    pred = model.predict(input_df)[0]
    return float(np.round(pred, 2))

def get_shap_explainer():
    return explainer

def get_expected_value():
    return expected_value

def get_clean_feature_names():
    return clean_feature_names
