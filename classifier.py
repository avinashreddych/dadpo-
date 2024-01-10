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


model_name = "avinashreddy/gpt-2-harmful"

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
        "padding": "max_length",
        "truncation": True,
        "max_length": 256,
        "return_tensors": "pt",
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

prepared_dataset = Dataset.from_csv("./data/harmless_test_data.csv")

formatted_dataset = prepared_dataset.map(formatting_func)

formatted_dataset = formatted_dataset.train_test_split()


# %%

training_args = TrainingArguments(
    output_dir="./research/reward_model",
    per_device_train_batch_size=24,
    gradient_accumulation_steps=4,
    learning_rate=1.41e-5,
    remove_unused_columns=False,
    optim="adamw_torch",
    logging_steps=1,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    run_name="phi-2-harm-finetuned",
    report_to=None,
)

trainer = ClassifierTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
)

# %%
trainer.train()


# reward model
model.save_pretrained("avinashreddy/reward_model")
tokenizer.save_pretrained("avinashreddy/reward_model")

# %%
