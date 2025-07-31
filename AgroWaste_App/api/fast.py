import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from AgroWaste_App.ml_logic.registry import load_model


app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(
        Moisture: str,
        Protein: float,
        Fat: float,
        Total_Carbohydrates	: float,
        Sugars: float,
        Dietary_Fiber: float,
        Crude_Fiber: float,
        Ash: float
    ):

    X_pred = pd.DataFrame([locals()])
    model = app.state.model
    y_pred = model.predict(X_pred)

    return {"FRAP_value": float(y_pred[0])}
