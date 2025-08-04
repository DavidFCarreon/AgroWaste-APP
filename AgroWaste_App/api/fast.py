import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from AgroWaste_App.ml_logic.registry import load_model
from AgroWaste_App.interface.chat_gpt_pipeline import obtain_features

app = FastAPI()
app.state.model = load_model()
app.state.gpt = obtain_features()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# http://127.0.0.1:8000/predict?Moisture=0.8&Protein=0.5&Fat=0.3&Total_Carbohydrates=0.7&Sugars=0.2&Dietary_Fiber=0.7&Crude_Fiber=0.1&Ash=0.8
@app.get("/predict")
def predict(
        Moisture: float,
        Protein: float,
        Fat: float,
        Total_Carbohydrates	: float,
        Sugars: float,
        Dietary_Fiber: float,
        Crude_Fiber: float,
        Ash: float
    ):
    X_pred = pd.DataFrame([locals()])
    X_pred = X_pred.rename(columns={
        "Total_Carbohydrates": "Total Carbohydrates",
        "Dietary_Fiber": "Dietary Fiber",
        "Crude_Fiber": "Crude Fiber"
    })

    model = app.state.model
    y_pred = model.predict(X_pred)

    return {"FRAP_value": float(y_pred[0])}

@app.get("/get_features")
def get_features(product_name: str
    ):

    model_gpt = app.state.gpt
    features = model_gpt.predict(product_name)

    X_pred = pd.DataFrame([features])
    model_get = app.state.model
    y_pred = model_get.predict(X_pred)

    ret=features
    ret["FRAP_value"] = float(y_pred[0])
    return ret
