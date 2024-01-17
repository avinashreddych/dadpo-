# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %pip install trl datasets

# !pip install datasets

import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler


import random
import pandas as pd
from operator import itemgetter
import torch
import warnings

warnings.filterwarnings("ignore")
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)
from trl import RewardTrainer


#read dataset
df_sub = pd.read_csv("./data/reward_train_data.csv")


# Select a base model whch we need to train for reward modeling.
# model_name = "distilroberta-base"
model_name="lvwerra/gpt2-imdb"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

def formatting_func(examples):
    kwargs = {
        "padding": "max_length",
        "truncation": True,
        "max_length": 512,
        "return_tensors": "pt",
    }
    prompt_plus_chosen_response = (
        examples["instruction"] + "\n" + examples["chosen_response"]
    )
    prompt_plus_rejected_response = (
        examples["instruction"] + "\n" + examples["rejected_response"]
    )
    tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0],
        "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0],
        "attention_mask_rejected": tokens_rejected["attention_mask"][0],
    }

rows = []
for record in df_sub.itertuples(index=True, name="Pandas"):
    if record is None or len(record) == 0:
        continue
    rows.append(
        {
            "instruction": record.query,
            "chosen_response": record.chosen,
            "rejected_response": record.rejected,
        }
    )

from datasets import Dataset
prepared_dataset = Dataset.from_list(rows)
prepared_dataset.to_pandas()

formatted_dataset = prepared_dataset.map(formatting_func)
formatted_dataset = formatted_dataset.train_test_split()

training_args = TrainingArguments(
    output_dir="./research/reward_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps = 16,
    learning_rate=1.41e-5,
    remove_unused_columns=False,
    optim="adamw_torch",
    logging_steps=1,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    report_to=None)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
)
trainer.train()


#reward model 
model.save_pretrained("gpt2_imdb_rwd_model_obj1")
tokenizer.save_pretrained("gpt2_imdb_rwd_tok_obj1")



