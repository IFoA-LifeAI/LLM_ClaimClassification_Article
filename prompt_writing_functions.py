from typing import List
import json


def write_initial_prompt(options: List[str]) -> str:
    """
    Python equivalent of write_initial_prompt() in R.
    """
    options_block = "\n".join(f"\"{opt}\"" for opt in options)
    return (
        "You are a classification LLM. You will receive a JSON file. "
        "The file will contain a list of items with cause_of_death.\n"
        "It is important that you return only an edited version of the JSON file. "
        "Add 'category' to each item, which can only ever pick one of the values below. "
        'If none are suitable choose the category of "none":\n\n'
        f"{options_block}\n\n"
        "No explanations. Return only the data in a structured JSON format. "
        "Your final JSON code must begin with ``` and end with ```.\n"
        "If a cause of death cannot be linked to smoking in any way, for example if it is an infectious disease, "
        "a genetic disorder, or has an external cause provided in the cause_of_death text (e.g. asbestos), "
        'then assign the category as "none".'
    )


def split_vector(vec: List[str], max_length_per_vec: int):
    """
    Python equivalent of split_vector():
    Split a list into chunks of length <= max_length_per_vec.
    Returns a generator of sublists.
    """
    for i in range(0, len(vec), max_length_per_vec):
        yield vec[i: i + max_length_per_vec]


def glue_to_json(vec: List[str]) -> str:
    return json.dumps([{"cause_of_death": v} for v in vec], ensure_ascii=False)

    """
    Python equivalent of glue_to_json():
    Convert a list of strings into a JSON-like string:
    [
      { "cause_of_death": "..." },
      { "cause_of_death": "..." },
      ...
    ]
    This returns a string (not a Python list/dict), matching the R behaviour.
    """
    """
    inner_lines = ",\n".join(
        f'  {{ "cause_of_death": "{v}" }}' for v in vec
    )
    return "[\n" + inner_lines + "\n]"
    """
