# %%
from LLMClassifier import LLMClassifier
import pandas as pd
import numpy as np

cod_df = pd.read_csv("./Data/cause_of_death.csv")
cod_list = cod_df["cause_of_death"][cod_df["train_test"] == "test"].to_list()

# %%

classifier = LLMClassifier(
    options=["ischaemic heart disease", "lung cancer", "stroke", "none"],
    input_field="cause_of_death",
    category_field="category",
    extra_kwargs={
        "logit_bias": {12851: -2.5},
        "logprobs": True
    }
)

final_df = classifier.run(cod_list, checkpoint_path="./Data/checkpoint.pkl")
final_df.to_csv("classified_data.csv", index=False)
