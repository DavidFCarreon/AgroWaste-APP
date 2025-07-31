import requests
import pandas as pd
import time
import os

# Inserta tu API KEY de FoodData Central
API_KEY = '-- API KEY --'
BASE_URL = "https://api.nal.usda.gov/fdc/v1"


# Mapa de columnas del DataFrame → nombre exacto en la respuesta de FDC
NUTRIENT_MAP = {
    "Water": "Water",
    "Nitrogen": "Nitrogen",
    "Ash": "Ash",
    "Carbohydrate, by summation": "Carbohydrate, by summation",
    "Carbohydrate, by difference": "Carbohydrate, by difference",
    "Sugars, Total": "Sugars, Total",
    "Total Sugars": "Total Sugars",
    "Protein": "Protein",
    "Total lipid (fat)": "Total lipid (fat)",
    "Fiber, total dietary": "Fiber, total dietary",
    "Energy": "Energy"
}


def get_food_details_by_id(fdc_id: int) -> dict:
    """Obtiene los datos completos de un alimento por su FDC ID."""
    url = f"{BASE_URL}/food/{fdc_id}"
    resp = requests.get(url, params={"api_key": API_KEY})
    resp.raise_for_status()
    return resp.json()

def extract_selected_nutrients(food_data: dict) -> dict:
    """Extrae los nutrientes definidos en NUTRIENT_MAP."""
    extracted = {col: None for col in NUTRIENT_MAP}
    for nut in food_data.get("foodNutrients", []):
        # Hay dos posibles estructuras: 'nutrient':{'name':...} o directamente 'nutrientName'
        name = nut.get("nutrient", {}).get("name") or nut.get("nutrientName", "")
        amount = nut.get("amount") or nut.get("value")
        # Compara nombre exacto
        for df_col, fdc_name in NUTRIENT_MAP.items():
            if name == fdc_name:
                extracted[df_col] = amount
    return extracted

def build_dataset_from_ids(allowed_categories: list, csv_path: str, id_column: str = "fdc_id") -> pd.DataFrame:
    """
    Lee un CSV con una columna de FDC IDs y devuelve un DataFrame con:
    - descripción
    - fdc_id
    - todos los nutrientes de NUTRIENT_MAP
    """
    df_ids = pd.read_csv(csv_path)
    records = []
    for fdc_id in df_ids[id_column].dropna().astype(int).unique():
        print(f"Procesando FDC ID: {fdc_id}")
        try:
            data = get_food_details_by_id(fdc_id)

            category = data.get("foodCategory", "")['description']
            if category not in allowed_categories:
                continue

            nutrients = extract_selected_nutrients(data)
            nutrients["fdc_id"] = fdc_id
            nutrients["description"] = data.get("description", "")
            nutrients['food']=data.get("inputFoods", "")[0].get("foodDescription", "")
            nutrients["category"] = category
            records.append(nutrients)
        except Exception as e:
            print(f"⚠️ Error con ID {fdc_id}: {e}")
        time.sleep(0.5)  # respetar límite de tasa
    return pd.DataFrame.from_records(records)




# 1) Asegúrate de tener un CSV 'fdc_ids.csv' con columna 'fdc_id'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the CSV file
csv_file = os.path.join(BASE_DIR, "fdc_ids.csv")
allowed_categories = ['Spices and Herbs','Fruits and Fruit Juices','Vegetables and Vegetable Products','Nut and Seed Products',
                          'Legumes and Legume Products']

# 2) Construir el DataFrame
df_proximal = build_dataset_from_ids(allowed_categories, csv_file, id_column="fdc_id")

# 3) Guardar a CSV
output_file = os.path.join(BASE_DIR, "fdc_ids_proximal_dataset.csv")

df_proximal['Sugars'] = df_proximal[['Sugars, Total', 'Total Sugars']].mean(axis=1, skipna=True)
df_final=df_proximal.drop(columns=['Sugars, Total','Total Sugars', 'category','fdc_id']).rename(columns={'description': 'Food Product', 'food': 'Side Stream'})
df_final.to_csv(output_file, index=False)
print(f"✅ Dataset final guardado en '{output_file}'")
print(df_proximal.head())
