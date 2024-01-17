# %%
from tqdm import tqdm
import pandas as pd
import torch

tqdm.pandas()

from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)


model_name = "avinashreddy/reward-model-gpt2"

model_name = "PKU-Alignment/beaver-7b-v1.0-reward"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.train(False)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
tokenizer.padding_side = "right"

# %%
from datasets import Dataset

# test_data = Dataset.from_csv("./data/reward_test_data-new.csv").shuffle(seed=42)

test_data = Dataset.from_csv("./harmful_generated_data.csv").shuffle(seed=42)


# %%
def prepare_sample_text(prompt, response):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {prompt}\n\nAnswer: {response}"
    return text


def formatting_func(sample):
    return {
        "chosen_text": prepare_sample_text(sample["prompt"], sample["chosen"]),
        "rejected_text": prepare_sample_text(sample["prompt"], sample["rejected"]),
    }


def harmful(sample):
    return sample["flag"] == "unsafe-unsafe"


def harmless(sample):
    return sample["flag"] != "unsafe-unsafe"


test_data = test_data.map(formatting_func)

# %%
kwargs = {
    "padding": "max_length",
    "truncation": True,
    "max_length": 256,
    "return_tensors": "pt",
}


# %%
def calculate_rlhf_vs_sft(sample):
    with torch.inference_mode():
        input_tokens = tokenizer(sample["sft_gpt_2_out"], **kwargs)
        sft_out_score = model(**input_tokens).logits.item()
        input_tokens = tokenizer(sample["rlhf_gpt2_out"], **kwargs)
        rlhf_out_score = model(**input_tokens).logits.item()
        return {
            "sft_out_score": sft_out_score,
            "rlhf_out_score": rlhf_out_score,
            "win": sft_out_score > rlhf_out_score,
        }


def scores(sample):
    # score_chosen = sentiment_pipe(prepare_sample_text(sample["prompt"], sample["chosen"]))[0]["score"]
    # score_rejected = sentiment_pipe(prepare_sample_text(sample["prompt"], sample["rejected"]))[0]["score"]
    with torch.inference_mode():
        input_tokens = tokenizer(
            prepare_sample_text(sample["prompt"], sample["chosen"]), **kwargs
        )
        score_chosen = model(**input_tokens).logits.item()
        input_tokens = tokenizer(
            prepare_sample_text(sample["prompt"], sample["rejected"]), **kwargs
        )
        score_rejected = model(**input_tokens).logits.item()
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
            "chosen_score": score_chosen,
            "rejected_score": score_rejected,
            "win": score_chosen > score_rejected,
        }


# %%
sft_rlhf_scores = test_data.map(calculate_rlhf_vs_sft)
sft_rlhf_scores.to_pandas()[
    ["sft_gpt_2_out", "rlhf_gpt2_out", "sft_out_score", "rlhf_out_score", "win"]
].to_csv("stats_with_harmfuL-data-pku.csv", index=False)

# %%
# sft_rlhf_scores = test_data.map(calculate_rlhf_vs_sft)
# sft_rlhf_scores.to_pandas()[
#     ["sft_gpt_2_out", "rlhf_gpt2_out", "sft_out_score", "rlhf_out_score", "win"]
# ].to_csv("stats_with_harmfuL-data.csv", index=False)

# %%
df = sft_rlhf_scores.to_pandas()

# %%
import matplotlib.pyplot as plt

plt.hist(df["sft_out_score"], label="sft")
plt.hist(df["rlhf_out_score"], label="rlhf")
plt.legend()
plt.show()


# # %%
# input_tokens = tokenizer(
#     prepare_sample_text(test_data[0]["prompt"], test_data[0]["rejected"]), **kwargs
# )

# out = model(**input_tokens)
# print(out.logits.item())

# # %%
# out.logits

# # %%
# sentiment_pipe = pipeline(
#     "sentiment-analysis", model="avinashreddy/reward-model", device="cuda", **kwargs
# )

# # %%
# sentiment_pipe(prepare_sample_text(test_data[0]["prompt"], test_data[0]["chosen"]))

# # %%
# import torch

# # %%
# with_scores_dataset = test_data.map(scores)

# # %%
# with_scores_dataset.filter(harmless).to_pandas().win.value_counts()

# # %%
# 89 / (89 + 266)

# # %%
# print(with_scores_dataset.to_pandas().to_markdown())

# # %%
# chosen_scores = sentiment_pipe(test_data["chosen_text"].tolist())
# chosen_df = pd.DataFrame(chosen_scores)
# test_data["chosen_score"] = chosen_df["score"]

# # %%
# rejected_scores = sentiment_pipe(test_data["rejected_text"].tolist())
# rejected_df = pd.DataFrame(rejected_scores)
# test_data["rejected_score"] = rejected_df["score"]

# # %%
# test_data

# # %%
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(-5, 5, len(test_data))

# # %%
# c_chosen = "blue"  # Color for chosen_score
# c_rejected = "red"  # Color for rejected_score

# plt.scatter(x, test_data["chosen_score"], color=c_chosen, label="Chosen Score")
# plt.scatter(x, test_data["rejected_score"], color=c_rejected, label="Rejected Score")

# # Add labels and title
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Scatter Plot with Colors")

# # Add legend
# plt.legend()

# # Show the plot
# plt.show()

# # %%
