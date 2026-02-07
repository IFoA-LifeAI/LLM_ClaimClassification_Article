# LLM_ClaimClassification_Article

Python code behind the article on LLM Classsification

The class `LLMClassifier` can be found in `LLMClassifier.py`. This is a versatile class for classifying datasets into disctinct categories. It also produces a confidence score for each guess in the output.

The `chat_gpt classify script.py` script gives a working example of how `LLMClassifier` can be imported and called to perform the classification task with a clean output.

The `fine-tuning script.py` creates OpenAI-compatible JSONL files for fine-tuning, uploads them to OpenAI, and creates a fine-tuned model. That model name can then be used in the `chat_gpt classify script.py` in the same way as a base model.
