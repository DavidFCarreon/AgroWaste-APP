import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Importing all datasets
df_bio_food = pd.read_csv(os.path.join(BASE_DIR, "BioFoodComp_cleaned_v2.csv"))
df_fdc = pd.read_csv(os.path.join(BASE_DIR, "./fdc/fdc_ids_proximal_dataset.csv"))
df_fwe = pd.read_csv(os.path.join(BASE_DIR, "./fwe/preprocessed.csv"))


PROXIMAL_MAP_BIO = {
    'Food name in English': 'Food Product',
    # Core Proximal Components
    'WATER(g)': 'Moisture',
    'PROTCNT(g)': 'Protein',
    'FATCE(g)': 'Fat',
    'CHOAVLDF(g)': 'Carbohydrates',
    'FIBTG(g)': 'Dietary Fiber',
    'ASH(g)': 'Ash',

    # Additional but common fields
    'SUGAR(g)': 'Sugars',
    'FIBND(g)': 'Insoluble Fiber',
    'FIBAD(g)': 'Soluble Fiber',

    # Less common / advanced or legacy fiber metrics
    'FIBC(g)': 'Crude Fiber',
    'CELLU(g)': 'Cellulose',
    'LIGN(g)': 'Lignin',
    'HEMCEL(g)': 'Hemicellulose'
}

df_bio_food.rename(columns=PROXIMAL_MAP_BIO, inplace=True)
df_bio_food.isna().sum()/df_bio_food.shape[0]*100


PROXIMAL_MAP_FDC = {
    'Water': 'Moisture',
    'Nitrogen': 'Nitrogen',
    'Ash': 'Ash',
    'Carbohydrate, by summation': 'Total Carbohydrates (sum)',
    'Carbohydrate, by difference': 'Total Carbohydrates',
    'Protein': 'Protein',
    'Total lipid (fat)': 'Fat',
    'Fiber, total dietary': 'Dietary Fiber',
    'Energy': 'Energy (kcal)',

    # Contextual fields
    'Food Product': 'Food Product',
    'Side Stream': 'Side Stream'
}

print(df_fdc.shape)
df_fdc.rename(columns=PROXIMAL_MAP_FDC, inplace=True)
df_fdc.drop(columns=['Nitrogen','Total Carbohydrates (sum)'], inplace=True)
df_fdc.isna().sum()/df_fdc.shape[0]*100


PROXIMAL_MAP_FWE = {
    'Food Product': 'Food Product',
    'Side Stream': 'Side Stream',
    # Core Proximal Components
    'Ash': 'Ash',
    'Dry Matter': 'Dry Matter',
    # Fiber-related components
    'Fibre, crude': 'Crude Fiber',
    'Acid Detergent Fibre (ADF)': 'ADF',
    'Cellulose': 'Cellulose',
    'Hemicellulose': 'Hemicellulose',
    'Pectin': 'Pectin',
    # Carbohydrates
    'Sugar, total': 'Sugars'
}

print(df_fwe.shape)
df_fwe.rename(columns=PROXIMAL_MAP_FWE, inplace=True)
df_fwe.isna().sum()/df_fwe.shape[0]*100

df_merged = pd.concat([df_bio_food, df_fdc], axis=0, ignore_index=True)
df_merged.to_csv(os.path.join(BASE_DIR, "Combined_BioFood_fdc.csv"), index=False)
