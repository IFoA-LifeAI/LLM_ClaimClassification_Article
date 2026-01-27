
# %%
from pathlib import Path
import pandas as pd
import numpy as np
import re

# inputs

seed = 1
test_n = 1000

filepaths = dict(
    all_cause_of_death_input=Path("./Data/Raw/34506561512084822.csv"),
    human_classification_input=Path("./Data/Raw/Human Classification v2.xlsx"),
    output=Path("./Data/cause_of_death.csv")
)


# import causes of death

def data_import_cod_vector(fp):
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


cod_vector = data_import_cod_vector(
    fp=filepaths.get("all_cause_of_death_input")
)

# read in human picks and attach

human_class_df = pd.read_excel(
    filepaths.get("human_classification_input")
).set_index("cause_of_death")

empty_picks = pd.DataFrame(
    dict(
        cause_of_death=pd.Series(cod_vector).str.lower(),
        category_human="none"
    )).set_index("cause_of_death")

human_class_df = human_class_df.combine_first(empty_picks)

# assign to train or test

tt = pd.Series(
    ["train"] * (len(human_class_df) - test_n) + ["test"] * test_n
).sample(
    n=len(human_class_df),
    replace=False,
    ignore_index=True,
    random_state=seed
)
tt.index = human_class_df.index
human_class_df["train_test"] = tt

# save output

human_class_df.to_csv(filepaths.get("output"))

# %%
