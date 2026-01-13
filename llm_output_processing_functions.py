import json
import re
import time
from typing import List, Optional

import pandas as pd


def json_list_to_df(output_list: List[str]) -> pd.DataFrame:
    """
     - Takes a list of raw LLM string responses
    - If there are ``` code fences, uses the LAST fenced block
    - Strips backticks and leading 'json'
    - Parses JSON into Python objects
    - Combines into a single pandas DataFrame
    """
    rows = []
    # i = 1
    for x in output_list:
        # print(i)
        # Check for ``` fenced blocks
        if "```" in x:
            matches = re.findall(r"```(.*?)```", x, flags=re.S)
            if matches:
                # Use the last fenced block
                x = matches[-1]

        # Remove backticks and leading 'json'
        x_clean = x.replace("`", "")
        x_clean = re.sub(r"^\s*json", "", x_clean, flags=re.I).strip()

        x_clean = re.sub(r",\s*}", "}", x_clean)
        x_clean = re.sub(r",\s*]", "]", x_clean)
        if not x_clean.endswith("]") and x_clean.strip().startswith("["):
            x_clean += "]"
        if not x_clean.endswith("}") and x_clean.strip().startswith("{"):
            x_clean += "}"

        try:
            parsed = json.loads(x_clean)
            if isinstance(parsed, dict):
                rows.append(parsed)
            elif isinstance(parsed, list):
                rows.extend(parsed)
            else:
                raise ValueError(
                    "Parsed JSON is neither a dict nor a list of dicts.")

        except json.JSONDecodeError as e:
            print("JSON parsing error:", e)
            print("Problematic string:\n", x_clean)
            continue

        # i = i + 1
    return pd.DataFrame(rows)


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
