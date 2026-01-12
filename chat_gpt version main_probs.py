# %%

# package imports
from openai import OpenAI
import time
import pickle
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# %%

# local imports
from prompt_writing_functions import split_vector, glue_to_json, write_initial_prompt
from data_import import data_import_cod_vector
from llm_output_processing_functions import json_list_to_df


# %%

# ---Update the name of data_import.py file; comment the one not in use---
# from data_import import load_cod_data

_ = load_dotenv()

client = OpenAI()     # uses OPENAI_API_KEY from .env

# set max chunk size... this might need refining based on what API seems to accept
max_chunk_size = 12
sleep_time_between_chunks = 0
model = "gpt-4o-2024-08-06"
output_name = "output_openai_gpt_4o_logprobs"
output_name_csv = "output_openai_gpt_4o_logprobs"

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

cod_vector = data_import_cod_vector(
    fp=Path("./Data/34506561512084822.csv")
)

# -------------------------------------------------------------------
# Prepare chunked prompts
# -------------------------------------------------------------------
list_x = list(split_vector(cod_vector, max_chunk_size))[:3]
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
        logprobs=True,
        temperature=0
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

# This function is making some huge assumptions about the structure so be careful!

# assumes...
# 1. each open curly brace is a new "guess"
# 2. the guesses start from the first curly brace
# 3. the logprobs inside the curly braces contains no noise (i.e. every token is added apart from curly brace themselves)

# ...and kind of nesting in JSON structure will break this


def extract_logprobs_assuming_json(token_data):

    # Build DataFrame
    df = pd.DataFrame({
        "token": [t.token for t in token_data],
        "logprob": [t.logprob for t in token_data],
    })

    df["open_brace_count"] = df["token"].str.count(r"\{")
    df["close_brace_count"] = df["token"].str.count(r"\}")
    df["open_brace_cum"] = df["open_brace_count"].cumsum()

    # trim down the df
    json_start = (df["open_brace_count"] > 0).values.argmax()
    is_close_brace = (df["close_brace_count"] > 0).values
    json_end = len(is_close_brace) - is_close_brace[::-1].argmax() - 1
    df = df.iloc[json_start:json_end,]

    # zeroise logprobs for curly brackets as they can false distort
    df["logprob"] = df["logprob"].where(
        (df["token"].str.count(r"\{") + df["token"].str.count(r"\}")) == 0,
        0
    )
    lp = df.groupby("open_brace_cum")["logprob"].sum("logprob").values
    return lp


# %%

# turn the message into dataframe
llm_output_message = [x.message.content for x in llm_output]
df = json_list_to_df(llm_output_message)

# get logprobs for each guess and add to dataframe
llm_output_logprobs = [x.logprobs.content for x in llm_output]
logprobs_arrays = [extract_logprobs_assuming_json(
    x) for x in llm_output_logprobs]
df["logprobs"] = np.concatenate(logprobs_arrays)

output_csv_path = Path("./Data") / f"{output_name_csv}.csv"
df.to_csv(output_csv_path)

# %%
