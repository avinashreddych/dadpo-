# %% [markdown]
#
# Training PHI-2 Model with harmfull responses dataset

# %%

from datasets import Dataset
from tqdm import tqdm
from trl.trainer import ConstantLengthDataset
import os

os.environ["WANDB_PROJECT"] = "phi2-harmful"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['prompt']}\n\nAnswer: {example['response']}"
    return text


def create_datasets(tokenizer, args):
    train_data = Dataset.from_csv("./train_data_pku.csv")
    valid_data = Dataset.from_csv("./test_data_pku.csv")
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


# %%
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    logging,
    set_seed,
)


@dataclass
class ScriptArgs:
    csv_path = "./harmfull_responses.csv"
    seed = 41
    seq_length = 256
    model = "gpt2"
    output_dir = "./output_dir/"
    max_steps = 10000
    batch_size = 24
    gradient_accumulation_steps = 4
    gradient_checkpointing = False
    learning_rate = 1e-4
    lr_scheduler_type = "cosine"
    num_warmup_steps = 100
    weight_decay = 0.05
    log_freq = 1
    eval_freq = 10
    save_freq = 500
    fp16 = False
    bf16 = False


args = ScriptArgs()

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
train_dataset, eval_dataset = create_datasets(tokenizer, args)

# %%



model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

# %%
training_args = TrainingArguments(
    output_dir=args.output_dir,
    dataloader_drop_last=True,
    evaluation_strategy="steps",
    max_steps=args.max_steps,
    eval_steps=args.eval_freq,
    save_steps=args.save_freq,
    logging_steps=args.log_freq,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_steps=args.num_warmup_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    fp16=args.fp16,
    bf16=args.bf16,
    weight_decay=args.weight_decay,
    run_name="phi-2-harm-finetuned",
    report_to="wandb",
    ddp_find_unused_parameters=False,
)

# %%
from trl import SFTTrainer

from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["Wqkv"],
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=None,
    packing=True,
    max_seq_length=256,
)

print_trainable_parameters(trainer.model)


# %%
trainer.train()

print("Saving last checkpoint of the model")
trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))

# %%

