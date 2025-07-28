import pandas as pd
import os
import openpyxl

#Get the base directory (where the script is located)
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', 'food_waste_explorer_jp')
# List all .xlsx files in the base directory
df= pd.read_excel(os.path.join(base_dir, 'Combined_foods_complete.xlsx'))

compounds= df['Compound'].value_counts()
side_streams = df['Side Stream'].value_counts()

with open('Compound_counts.txt', 'w') as f:
    for name, count in compounds.items():
        f.write(f"{name}: {count}\n")

with open('Side_Stream_counts.txt', 'w') as f:
    for name, count in side_streams.items():
        f.write(f"{name}: {count}\n")


import string
import numpy as np

def cleaning(sentence):
    # Handle missing or non-string input
    if not isinstance(sentence, str):
        return np.nan

    sentence = sentence.strip().lower()  # basic cleaning

    # Remove punctuation except dash "-" because we need it for ranges like "10-12"
    punctuation_without_dash = string.punctuation.replace('-', '')
    for p in punctuation_without_dash:
        sentence = sentence.replace(p, '')

    # Now handle numeric ranges like "10-12"
    # If sentence is a range like '10-12', return the mean 11
    if '-' in sentence:
        parts = sentence.split('-')
        try:
            nums = [float(p) for p in parts]
            mean_val = sum(nums) / len(nums)
            return mean_val
        except ValueError:
            # if conversion fails, return sentence as is (or np.nan)
            return np.nan

    # If sentence is just a number string, convert to float
    try:
        return float(sentence)
    except ValueError:
        # if it's not a number or a range, return np.nan or original cleaned string
        return np.nan


df['clean_level'] = df['Level'].apply(cleaning)


df.columns = df.columns.str.strip()  # Clean column names if needed
compounds_of_interest = ['Ash', 'Dry Matter', 'Protein, crude'
                        'Fat, crude', 'Acid Detergent Fibre (ADF)', 'Fibre, crude', 'Neutral Detergent Fibre'
                        'Lignin', 'Cellulose', 'Hemicellulose', 'Nitrogen, total'
                        'Sugar', 'Sugar, total','Pectin']

# Filter to only compounds in your list
filtered_df = df[df['Compound'].isin(compounds_of_interest)]

# Pivot
pivot = filtered_df.pivot_table(
    index=['Food Product', 'Side Stream'],
    columns='Compound',
    values='clean_level',
    aggfunc='sum'
).reset_index()


pivot.to_csv('pivot_output.csv', index=False)
