# %%
import pandas as pd
import numpy as np
from pathlib import Path
import great_tables as gt
from plotnine import *
np.random.seed(123)

# %%

# Human predictions
human_pred_csv = Path("./Data/cause_of_death.csv")
cod_df = pd.read_csv(human_pred_csv)
hm_df = cod_df[cod_df["train_test"] == "test"][[
    "cause_of_death", "category_human"]]


# %%
# AI predictions with logprobs
llm_pred_csv = "output_openai_gpt_4o_logprobs"
output_csv_path = Path("./Data/output_gpt_4o_base.csv")
lp_df = pd.read_csv(output_csv_path)

cols_keep = ["cause_of_death", "category", "confidence"]
lp_df = lp_df.loc[:, cols_keep]

# %%

rec = pd.merge(hm_df, lp_df, on="cause_of_death")
rec["match"] = rec["category_human"] == rec["category"]


# %%
(ggplot(rec, aes(y="confidence", x="match", color="match")) +
    geom_jitter(width=0.3, alpha=0.35) +
    coord_flip() +
    scale_y_reverse() +
    xlab("LLM Guess Matches Human") +
    ylab("Confidence of Guess") +
    scale_color_manual(values={True: "forestgreen", False: "blue"}) +
    theme_bw() +
    theme(
        legend_position="none",
        axis_line=element_blank(),      # removes axis lines (flip-safe)
        panel_border=element_blank(),   # removes plot outline
        panel_grid=element_blank()
)
)

# %%
rec["confidence_band"] = pd.cut(rec["confidence"], bins=[0, 0.99, 1.0])

accuracy_df = rec.groupby("confidence_band", observed=False)[
    "match"].aggregate(["count", "sum", "mean"])

accuracy_gt = gt.GT(accuracy_df.reset_index())
accuracy_gt = accuracy_gt.fmt_percent(["mean"], decimals=1)

(accuracy_gt.cols_label(
    {
        "confidence_band": "Probability",
        "count": "Attempts",
        "sum": "Matches",
        "mean": "Accuracy"
    }
)
)
# %%
