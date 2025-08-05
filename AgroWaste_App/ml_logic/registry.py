import joblib

def load_model():
    model = joblib.load("AgroWaste_App/api/final_model.pkl")

    return model

def load_explainer():
    model = joblib.load("AgroWaste_App/api/explainer.pkl")

    return model
