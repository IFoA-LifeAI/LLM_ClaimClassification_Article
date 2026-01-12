# %%
import pandas as pd
import numpy as np
import re
from pathlib import Path


def data_import_cod_vector(fp=Path("./Data/34506561512084822.csv")):
    # Set random seed
    np.random.seed(1)

    # Read CSV file, skipping first 10 rows
    data = pd.read_csv(fp, skiprows=10)

    # Select rows 2 to 8225 (Python uses 0-based indexing, so this is rows 1:8225)
    data = data.iloc[1:8225]

    # Clean column names (lowercase, replace spaces with underscores)
    data.columns = data.columns.str.lower().str.replace(
        ' ', '_').str.replace('[^a-z0-9_]', '', regex=True)

    # Calculate total across all columns except 'cause_of_death'
    cod_data = data.copy()
    numeric_cols = [col for col in cod_data.columns if col != 'cause_of_death']
    cod_data[numeric_cols] = cod_data[numeric_cols].astype("int")

    cod_data['total'] = cod_data[numeric_cols].sum(axis=1)

    # Select only cause_of_death and total columns
    cod_data = cod_data[['cause_of_death', 'total']]
    cod_data = cod_data.rename(
        columns={'cause_of_death': 'cause_of_death_full'})

    # Filter rows where total > 0
    cod_data = cod_data[cod_data['total'] > 0].copy()

    # Extract letter (first character if it's A-Z)
    cod_data['letter'] = cod_data['cause_of_death_full'].str.extract(
        r'^([A-Z])', expand=False)

    # Extract cause of death text (from position 7 onwards, remove )(:, characters)
    cod_data['cause_of_death'] = (cod_data['cause_of_death_full']
                                  # Python uses 0-based indexing, so position 7 is index 6
                                  .str[6:]
                                  .str.replace(r'[)(:,]', '', regex=True))

    cod_vector = cod_data['cause_of_death'].unique()

    # Shuffle the cod_vector ...just to make it harder for the robots :)
    cod_vector = np.random.choice(
        cod_vector, size=len(cod_vector), replace=False)

    return cod_vector
