import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import Dataset


lm_model_name= "avinashreddy/gpt-2-harmful"
reward_model_name = "avinashreddy/reward-model-gpt2"


#config
config = PPOConfig(
    model_name=lm_model_name,
    learning_rate=1.41e-5,
    log_with="wandb",
    seed=42,
    batch_size=16,
    gradient_accumulation_steps=4
        
)

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}


def prepare_sample_text(prompt, response):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {prompt}\n\nAnswer: {response}"
    return text




def formatting_func(sample):
    sample["input_ids"] = tokenizer.encode(prepare_sample_text(sample["prompt"] , ""))
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample
    

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

train_dataset_name = "reward_train_data-new.csv"
test_dataset_name = "reward_test_data-new.csv"
train_dataset = Dataset.from_csv("./data/" + train_dataset_name)
eval_dataset = Dataset.from_csv("./data/" + test_dataset_name)

train_dataset = train_dataset.map(formatting_func)
test_dataset = eval_dataset.map(formatting_func)

train_dataset.set_format(type ="torch")
test_dataset.set_format(type= "torch")




def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

#load the policy models
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

#tokenizer
tokenizer.pad_token = tokenizer.eos_token
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=train_dataset, data_collator=collator)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"

# reward_model_path = '/content/drive/MyDrive/gpt2_imdb_rwd_tok_50_50'
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
reward_model_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model_tokenizer.pad_token = reward_model_tokenizer.eos_token
gen_kwargs = {"min_length": -1, "top_k": 0, "top_p": 1, "do_sample": True, "temperature" : 0.5,"pad_token_id": tokenizer.eos_token_id}


generation_kwargs = {
    "min_length": -1,
    "top_k": 0,
    "top_p": 1,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "temperature" : 0.5,
    "max_length" : 256
}



for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # print(query_tensors)
    
    #### Get response from gpt2
    response_tensors = []
    for query in tqdm(query_tensors):
        # print("query", query)
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-len(query):])
        # response_tensors.append(response.squeeze())
        # print("response", response)
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # print(batch["response"])

    ### Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    texts_tokens = reward_model_tokenizer(batch["response"], return_tensors='pt', padding=True)
    outputs = reward_model(**texts_tokens)
    rewards = list(outputs.logits)

    # #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

print('done')

#save
ppo_trainer.model.save_pretrained('avinashreddy/gpt-2-rlhf-finetuned')