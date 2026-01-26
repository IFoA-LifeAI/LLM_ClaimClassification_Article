# %%
# -------------------------------------------------------------------
# Special section to find out token number of "none"
# This part is not 100% essential for code beyond
# -------------------------------------------------------------------
import tiktoken

model = "gpt-4o-2024-08-06"
encoding = tiktoken.encoding_for_model(model)
options = "none"
tokens = encoding.encode(options)  # 12851 is the code for "none"
tokens

# %%
