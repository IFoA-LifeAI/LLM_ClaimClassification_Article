# LLMClassifier

Python code behind the article "Machine Clean: how to use LLMs to wrangle messy datasets" submitted to The Actuary (found here: tbc).

If you are interested in  using the techniques in the article, the class `LLMClassifier` can be used. This is a versatile class for classifying datasets into disctinct categories. It also produces a confidence score for each guess in the output. This is found in `LLMClassifier.py`.

The `chat_gpt classify script.py` script gives a working example of how `LLMClassifier` can be imported and called to perform the classification task with a clean output. 

The `fine-tuning script.py` creates OpenAI-compatible JSONL files for fine-tuning, uploads them to OpenAI, and creates a fine-tuned model. That model name can then be used in the `chat_gpt classify script.py` in the same way as a base model.

All work was completed by Paul Beard and Kay Khine Myo.