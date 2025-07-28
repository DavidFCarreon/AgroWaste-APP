import pandas as pd
import os

# Get the base directory (where the script is located)
base_dir = os.path.dirname(os.path.abspath(__file__))

# List all .xlsx files in the base directory
excel_files = [f for f in os.listdir(base_dir) if f.endswith('.xlsx')]

# Read each .xlsx file into a DataFrame
dfs = [pd.read_excel(os.path.join(base_dir, f)) for f in excel_files]

# Concatenate all DataFrames into one
df = pd.concat(dfs, ignore_index=True)

df_cleaned=df[df['Compound Group']=='Proximates'].reset_index(drop=True)

df_cleaned.to_excel("Combined_foods2.xlsx", index=False)
