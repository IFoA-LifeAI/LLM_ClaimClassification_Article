# %% Imports
from LLMClassifier import LLMClassifier
import pandas as pd
import numpy as np

# %% Basic data prep
cod_df = pd.read_csv("./Data/cause_of_death.csv")
cod_list = cod_df["cause_of_death"][cod_df["train_test"] == "test"].to_list()
cod_categories = [
    "ischaemic heart disease",
    "cerebrovascular disease",
    "pulmonary disease",
    "lung cancer",
    "colorectal cancer",
    "larynx cancer",
    "kidney cancer",
    "acute myeloid leukemia",
    "oral cavity cancer",
    "esophageal cancer",
    "pancreatic cancer",
    "bladder cancer",
    "stomach cancer",
    "prostate cancer",
    "none"]


# %% Execute LLMClassifier and save
filename_out = "output_gpt_4o_base"

classifier = LLMClassifier(
    options=cod_categories,
    input_field="cause_of_death",
    category_field="category",
    model="gpt-4o-2024-08-06",
    extra_kwargs={
        # "logit_bias": {12851: -2.5}
    }
)

final_df = classifier.run(
    cod_list,
    chunk_size=100,
    checkpoint_path=f"./Data/{filename_out}.pkl",
    extra_info="If a cause of death cannot be linked to smoking in any way, for example if it is an infectious disease,"
    "a genetic disorder, or has an external cause provided within the text (e.g. asbestos),"
    "then assign the category as 'none'"
)

final_df.to_csv(f"./Data/{filename_out}.csv", index=False)
