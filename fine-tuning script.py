# -*- coding: utf-8 -*-

#%% Imports

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from openai import OpenAI
import time

#%% Convert dataframe into OpenAI fine-tuning JSONL format   
def build_jsonl(
    df,
    output_path,
    input_field = "cause_of_death",
    category_field = "category",
    chunk_size = 15,
    extra_info = None):

    system_message = (
        f"You are a medical cause-of-death classification model. "
        f"Classify the provided {input_field} list into the {category_field} categories. "
        f"Return only valid JSON matching the schema. "
        f"Do not invent categories.")

    if extra_info:
        system_message += " " + extra_info

    rows = df.sample(frac=1, random_state=1).to_dict("records")

    with open(output_path, "w") as f:
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i + chunk_size]

            record = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": json.dumps({
                        "items": [{input_field: r[input_field]} for r in chunk]
                    })},
                    {"role": "assistant", "content": json.dumps({
                        "items": [
                            {input_field: r[input_field],
                             category_field: r[category_field]}
                            for r in chunk]
                    })}
                ]
            }

            f.write(json.dumps(record) + "\n")

# Load dataset
cod_df = pd.read_csv("./Data/cause_of_death.csv")
cod_train_df = (cod_df
                .loc[cod_df["train_test"] == "train", ["cause_of_death", "category_human"]]
                .rename(columns={"category_human": "category"}))

# Remove rare classes (<2) before stratified split
vc = cod_train_df["category"].value_counts()
rare_classes = vc[vc < 2].index

rare_df = cod_train_df[cod_train_df["category"].isin(rare_classes)]
main_df = cod_train_df[~cod_train_df["category"].isin(rare_classes)]

train_df, val_df = train_test_split(main_df, test_size=0.2, stratify=main_df["category"], random_state=1)

# Add rare classes back into training only
train_df = pd.concat([train_df, rare_df], ignore_index=True)

extra_rules = (
    "If a cause of death cannot be linked to smoking in any way, "
    "for example if it is an infectious disease, a genetic disorder, "
    "or has an external cause provided within the text (e.g. asbestos), "
    "then assign the category as 'none'.")

build_jsonl(train_df, "./Data/train.jsonl", extra_info=extra_rules)
build_jsonl(val_df, "./Data/validation.jsonl", extra_info=extra_rules)

#%% Upload files
client = OpenAI()

train_file = client.files.create(file=open("./Data/train.jsonl", "rb"), purpose="fine-tune")
val_file = client.files.create(file=open("./Data/validation.jsonl", "rb"), purpose="fine-tune")

#%% Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=val_file.id,
    model="gpt-4o-mini-2024-07-18",
    seed=1,
    method={
        "type": "supervised",
        "supervised": {
            "hyperparameters": {
                "n_epochs": 3,
                "learning_rate_multiplier": 0.8,
                "batch_size": "auto"}}})

print("Fine-tuning job ID:", job.id)

while True:
    status = client.fine_tuning.jobs.retrieve(job.id)
    print("Status:", status.status)

    if status.status in ["succeeded", "failed"]:
        break

    time.sleep(30)

if status.status == "succeeded":
    print("Fine-tuned model:", status.fine_tuned_model)

