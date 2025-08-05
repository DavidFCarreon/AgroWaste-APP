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