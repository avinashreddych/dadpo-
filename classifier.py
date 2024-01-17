# %%
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from datasets import Dataset

import random
import pandas as pd
from operator import itemgetter
import torch
import warnings


from classifier_trainer import ClassifierTrainer


# %%
import os

os.environ["WANDB_PROJECT"] = "reward-model"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

model_name = "gpt2"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
tokenizer.padding_side = "right"


def prepare_sample_text(prompt, response):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {prompt}\n\nAnswer: {response}"
    return text


def formatting_func(examples):
    kwargs = {
        "truncation": True,
        "max_length": 256,
        "return_tensors": "pt",
        "padding" : "max_length"
    }
    prompt_plus_chosen_response = prepare_sample_text(
        examples["prompt"], examples["chosen"]
    )
    prompt_plus_rejected_response = prepare_sample_text(
        examples["prompt"], examples["rejected"]
    )
    tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0],
        "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0],
        "attention_mask_rejected": tokens_rejected["attention_mask"][0],
    }


# %%

train_dataset_name = "reward_train_data.csv"
test_dataset_name = "reward_test_data.csv"
train_dataset = Dataset.from_csv("./data/" + train_dataset_name)
eval_dataset = Dataset.from_csv("./data/" + test_dataset_name)

train_dataset = train_dataset.map(formatting_func)
test_dataset = eval_dataset.map(formatting_func)



# %%

training_args = TrainingArguments(
    output_dir="./research/",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=1.41e-5,
    remove_unused_columns=False,
    optim="adamw_torch",
    logging_steps=1,
    max_steps=400,
    num_train_epochs=5,
    evaluation_strategy="steps",
    run_name=f"{train_dataset_name.split('_')[0]}-model",
    report_to="wandb",
    save_strategy="steps",
    save_steps=50,
    save_total_limit=6,
    eval_steps=50,
    load_best_model_at_end = True
)

trainer = ClassifierTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    max_length=256,
)

# %%
trainer.train()


# reward model
model.save_pretrained(f"avinashreddy/{train_dataset_name.split('_')[0]}-model-gpt2")
tokenizer.save_pretrained(f"avinashreddy/{train_dataset_name.split('_')[0]}-model-gpt2")

# %%
