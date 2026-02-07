# %%
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from openai import OpenAI


class LLMClassifier:
    """
    A zero-shot text classifier that leverages OpenAI's Structured Outputs (JSON Schema)
    to categorize data while extracting confidence scores from logprobs.
    """

    def __init__(
        self,
        options: List[str],
        input_field: str = "text_input",
        category_field: str = "category",
        model: str = "ft:gpt-4o-mini-2024-07-18:personal::D4ZkhNZ6",
        seed: int = 1, 
        extra_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the classifier with schema definitions and API settings.

        Args:
            options: List of valid category labels.
            input_field: The key name for the input text in the JSON schema.
            category_field: The key name for the predicted label in the JSON schema.
            model: OpenAI model string (must support Structured Outputs).
            seed:Random seed for reproducible model outputs.
            extra_kwargs: Additional parameters for the ChatCompletion call (e.g., max_tokens).
        """
        self.client = OpenAI()
        self.options = options if "none" in options else options + ["none"]
        self.input_field = input_field
        self.category_field = category_field
        self.model = model
        self.seed = seed

        # Merge default settings safely with your extra_kwargs
        self.kwargs = {"temperature": 0, "logprobs": True, "seed": self.seed}
        if extra_kwargs:
            self.kwargs.update(extra_kwargs)

        # Fixed JSON Schema: Forces the LLM to return exactly what we need
        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "classification_results",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    self.input_field: {"type": "string"},
                                    self.category_field: {
                                        "type": "string", "enum": self.options}
                                },
                                "required": [self.input_field, self.category_field],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["items"],
                    "additionalProperties": False
                }
            }
        }

    def _extract_label_logprobs(self, logprob_content: List[Any], expected_count: int) -> np.ndarray:
        """
        Sums logprobs per JSON object using brace counting. 
        Aligns the last N groups with the expected row count.
        """
        df = pd.DataFrame({
            "token": [t.token for t in logprob_content],
            "logprob": [t.logprob for t in logprob_content],
        })

        # Count opening braces to group tokens by JSON object
        df["group_id"] = df["token"].str.count(r"\{").cumsum()
        group_sums = df.groupby("group_id")["logprob"].sum().tolist()

        # Structured output root is always group 1, so items are the trailing groups
        if len(group_sums) >= expected_count:
            return np.array(group_sums[-expected_count:])

        return np.pad(group_sums, (0, max(0, expected_count - len(group_sums))), constant_values=np.nan)

    def run(self,
            data_list: List[str],
            chunk_size: int = 15,
            checkpoint_path: Optional[str] = None,
            extra_info: Optional[str] = None) -> pd.DataFrame:
        """
        Main execution loop.
        :param extra_info: Additional context or rules for the system prompt.
        """
        all_responses = []

        # Construct the system message with optional extra info
        sys_msg = f"Classify the provided {self.input_field} list into the {self.category_field} categories."
        if extra_info:
            sys_msg += f" {extra_info}"

        for i in range(0, len(data_list), chunk_size):
            chunk = data_list[i: i + chunk_size]
            payload = json.dumps(
                {"items": [{self.input_field: item} for item in chunk]})

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": payload},
                ],
                response_format=self.response_format,
                **self.kwargs
            )

            all_responses.append(resp.choices[0])
            print(
                f"Completed chunk {i//chunk_size + 1} of {int(np.ceil(len(data_list)/chunk_size))}")

            if checkpoint_path:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(all_responses, f)

        return self._assemble(all_responses)

    def _assemble(self, responses) -> pd.DataFrame:
        chunk_dfs = []
        for res in responses:
            data = json.loads(res.message.content)
            df = pd.DataFrame(data["items"])

            if self.kwargs.get("logprobs") and not df.empty:
                confidence_logprobs = self._extract_label_logprobs(
                    res.logprobs.content, expected_count=len(df)
                )
                df["confidence"] = np.exp(confidence_logprobs)
            chunk_dfs.append(df)

        return pd.concat(chunk_dfs, ignore_index=True)
