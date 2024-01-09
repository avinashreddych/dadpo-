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


#config
config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    learning_rate=1.41e-5
)

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}


#Dataset processing
# Used for getting the dataset from csv
def build_dataset_new(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    
    #load the dataset
    df = pd.read_csv('/cmlscratch/schakra3/dpo_repo/pref_obj1.csv')
    df.rename(columns={"query": "prompt"}, inplace=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = Dataset.from_pandas(df)
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# Used for getting the dataset from csv
dataset = build_dataset_new(config)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

#load the policy models
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

#tokenizer
tokenizer.pad_token = tokenizer.eos_token
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"

# reward_model_path = '/content/drive/MyDrive/gpt2_imdb_rwd_tok_50_50'
reward_model = AutoModelForSequenceClassification.from_pretrained('/cmlscratch/schakra3/dpo_repo/gpt2_imdb_rwd_model_obj1')
reward_model_tokenizer = AutoTokenizer.from_pretrained('/cmlscratch/schakra3/dpo_repo/gpt2_imdb_rwd_tok_obj1')
reward_model_tokenizer.pad_token = reward_model_tokenizer.eos_token
gen_kwargs = {"min_length": -1, "top_k": 0, "top_p": 1, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}


generation_kwargs = {
    "min_length": -1,
    "top_k": 0,
    "top_p": 1,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

output_min_length = 40
output_max_length = 60
output_length_sampler = LengthSampler(output_min_length, output_max_length)


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    #### Get response from gpt2
    response_tensors = []
    for query in tqdm(query_tensors):
        gen_len = 45
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    texts_tokens = reward_model_tokenizer(texts, return_tensors='pt', padding=True)
    outputs = reward_model(**texts_tokens)
    rewards = list(outputs.logits)

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

print('done')

#save
ppo_trainer.model.save_pretrained('/cmlscratch/schakra3/dpo_repo/rlhf_model_obj1')