import joblib

def load_model():
    model = joblib.load("AgroWaste_App/api/best_model.pkl")

    return model
