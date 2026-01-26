# %%

# package imports
from openai import OpenAI
import time
import pickle
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# local imports
from prompt_writing_functions import split_vector, glue_to_json, write_initial_prompt
from llm_output_processing_functions import json_list_to_df, extract_logprobs_assuming_json


# %%
# -------------------------------------------------------------------
# Inputs / Run Settings
# -------------------------------------------------------------------

client = OpenAI()     # uses OPENAI_API_KEY from .env

seed = 1

# set max chunk size... this might need refining based on what API seems to accept
max_chunk_size = 12
sleep_time_between_chunks = 0
model = "gpt-4o-2024-08-06"
output_name = "output_openai_gpt_4o_test"
output_name_csv = "output_openai_gpt_4o_test"

extra_llm_create_kwargs = {
    "logprobs": True,
    "temperature": 0,
    "logit_bias": {
        12851: 100
    }  # <-- setting token "none" to be always picked!
}


# the options are taken from the following paper...
# https://pmc.ncbi.nlm.nih.gov/articles/PMC3229033/#:~:text=Current%20smokers%20had%20significantly%20higher,23.93)%2C%20smoking%2Drelated%20cancers
options = [
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
    "none",
]

# %%
cod = pd.read_csv(Path("./Data/cause_of_death.csv"))
cod_vector = np.array(
    cod[cod["train_test"] == "train"]["cause_of_death"]
)
rng = np.random.default_rng(seed=seed)
cod_vector = rng.permutation(cod_vector)

# %%

# -------------------------------------------------------------------
# Prepare chunked prompts
# -------------------------------------------------------------------
list_x = list(split_vector(cod_vector, max_chunk_size))
prompts_list = [glue_to_json(chunk) for chunk in list_x]
vectors = range(len(prompts_list))

# create an output list same length
llm_output = [None] * len(prompts_list)

# output RDS equivalent
output_path = Path("./Data") / f"{output_name}.pkl"

# %%

# -------------------------------------------------------------------
# Loop through chunks and call OpenAI
# -------------------------------------------------------------------
for i in vectors:

    system_prompt = write_initial_prompt(options)

    # Call OpenAI model
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts_list[i]},
        ],
        **extra_llm_create_kwargs
    )

    # Extract response text
    llm_output[i] = resp.choices[0]

    print(f"completed {i+1} of {len(vectors)}")

    # Save intermediate results
    with open(output_path, "wb") as f:
        pickle.dump(llm_output, f)

    # same as Sys.sleep()
    time.sleep(sleep_time_between_chunks)

# -------------------------------------------------------------------
# The final output as a single DataFrame:
# -------------------------------------------------------------------

# %%

# turn the message into dataframe
llm_output_message = [x.message.content for x in llm_output]
df = json_list_to_df(llm_output_message)

# get logprobs for each guess and add to dataframe
if extra_llm_create_kwargs.get("logprobs", default=False):
    llm_output_logprobs = [x.logprobs.content for x in llm_output]
    logprobs_arrays = [extract_logprobs_assuming_json(
        x) for x in llm_output_logprobs]
    df["logprobs"] = np.concatenate(logprobs_arrays)

output_csv_path = Path("./Data") / f"{output_name_csv}.csv"
df.to_csv(output_csv_path)

# %%
