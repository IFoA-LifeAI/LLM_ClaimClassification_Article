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
