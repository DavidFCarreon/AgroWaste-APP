import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from AgroWaste_App.ml_logic.registry import load_model, load_explainer
from AgroWaste_App.interface.chat_gpt_pipeline import obtain_features,obtain_comments
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))




app = FastAPI()
app.state.model = load_model()
app.state.explainer = load_explainer()


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
        'Moisture': 'moisture',
        'Protein': 'protein',
        'Fat': 'fat',
        'Ash': 'ash',
        "Sugars": "sugars",
        "Total_Carbohydrates": "total_carbohydrates",
        "Dietary_Fiber": "dietary_fiber",
        "Crude_Fiber": "crude_fiber"
    })

    model = app.state.model
    y_pred = model.predict(X_pred)

    explainer= app.state.explainer
    shap_values = explainer(X_pred)

    return {"FRAP_value": float(y_pred[0]),'shap_values': shap_values.values.tolist(),
            'shap_base_values': shap_values.base_values.tolist(),'shap_data': shap_values.data.tolist()}

# http://127.0.0.1:8000/get_features?product_name=banana
@app.get("/get_features")
def get_features(product_name: str
    ):

    features = obtain_features(product_name)

    X_pred = pd.DataFrame([features])
    X_pred = X_pred.rename(columns={
        'Moisture': 'moisture',
        'Protein': 'protein',
        'Fat': 'fat',
        'Ash': 'ash',
        "Sugars": "sugars",
        "Total_Carbohydrates": "total_carbohydrates",
        "Dietary_Fiber": "dietary_fiber",
        "Crude_Fiber": "crude_fiber"
    })

    model_get = app.state.model
    y_pred = model_get.predict(X_pred)

    explainer= app.state.explainer
    shap_values = explainer(X_pred)

    ret=features
    ret["FRAP_value"] = float(y_pred[0])
    ret['shap_values']= shap_values.values.tolist()
    ret['shap_base_values']= shap_values.base_values.tolist()
    ret['shap_data']= shap_values.data.tolist()
    return ret

# http://127.0.0.1:8000/get_comments?FRAP_value=30&product_name=banana
@app.get("/get_comments")
def comments(FRAP_value: float, product_name: str):

    comments_ = obtain_comments(FRAP_value, product_name)

    return {"Comments": comments_}
