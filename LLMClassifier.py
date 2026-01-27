# %%
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from openai import OpenAI


class LLMClassifier:
    def __init__(
        self,
        options: List[str],
        input_field: str = "text_input",
        category_field: str = "category",
        model: str = "gpt-4o-2024-08-06",
        extra_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.client = OpenAI()
        self.options = options if "none" in options else options + ["none"]
        self.input_field = input_field
        self.category_field = category_field
        self.model = model

        # Merge default settings safely with your extra_kwargs
        self.kwargs = {"temperature": 0, "logprobs": True}
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
        Uses cumulative brace counting to sum logprobs for each JSON object.
        Skips the root wrapper and aligns directly with the parsed items.
        """
        df = pd.DataFrame({
            "token": [t.token for t in logprob_content],
            "logprob": [t.logprob for t in logprob_content],
        })

        # Every '{' starts a new group.
        # Group 1 = {"items": [
        # Group 2 = { "input": ..., "category": ... }
        df["group_id"] = df["token"].str.count(r"\{").cumsum()

        # Sum logprobs by group
        # Since structural tokens are ~0 logprob, they won't dilute the 'decision' token
        group_sums = df.groupby("group_id")["logprob"].sum().tolist()

        # Alignment logic: The items are always the LAST 'expected_count' groups
        if len(group_sums) > expected_count:
            item_logprobs = group_sums[-expected_count:]
        else:
            # Fallback for unexpected tokenization
            item_logprobs = np.pad(group_sums, (0, max(
                0, expected_count - len(group_sums))), constant_values=np.nan)

        return np.array(item_logprobs)

    def run(self, data_list: List[str], chunk_size: int = 12, checkpoint_path: Optional[str] = None) -> pd.DataFrame:
        """Main execution loop with chunking and Structured Output."""
        all_responses = []
        sys_msg = f"Classify the provided {self.input_field} list into the {self.category_field} categories."

        for i in range(0, len(data_list), chunk_size):
            chunk = data_list[i: i + chunk_size]
            payload = json.dumps(
                {"items": [{self.input_field: item} for item in chunk]})

            # Pass all arguments, ensuring logprobs is only passed once
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
        """Parses the raw API responses into a single combined DataFrame."""
        chunk_dfs = []
        for res in responses:
            # Structured output ensures valid JSON, no cleaning needed
            data = json.loads(res.message.content)
            df = pd.DataFrame(data["items"])

            if self.kwargs.get("logprobs") and not df.empty:
                confidence_logprobs = self._extract_label_logprobs(
                    res.logprobs.content, expected_count=len(df)
                )
                df["confidence"] = np.exp(confidence_logprobs)
            chunk_dfs.append(df)

        return pd.concat(chunk_dfs, ignore_index=True)
