import tyro
import numpy as np
import json
import multiprocessing
import requests
import os
from tqdm import tqdm
from rich import print
from rich.prompt import Prompt
from rich.console import Console
from dataclasses import dataclass, field
from transformers import set_seed
from tqdm import tqdm
from datetime import datetime
from functools import partial
from datasets import load_dataset, concatenate_datasets

@dataclass(kw_only=True)
class ScriptArguments:
    endpoint_url: str = field(default="http://10.140.1.178:33998/v1")
    model_name: str = field(default="mistral", metadata={"help": '["Qwen1.5-72B-Chat", "Llama3-70B", "Llama3-70B-Instruct"]'})
    dataset_lists: tuple[str, ...] = field(default_factory=lambda: [
        # "/mnt/petrelfs/wangyuanfu/workspace/LLaMA-Factory/data/custom_dataset/helpsteer.py",
        ["/mnt/petrelfs/wangyuanfu/workspace/LLaMA-Factory/data/custom_dataset/hh_rlhf.py", "harmless"]
        # "/mnt/petrelfs/wangyuanfu/workspace/LLaMA-Factory/data/custom_dataset/safe_rlhf.py",
        # "/mnt/petrelfs/wangyuanfu/workspace/LLaMA-Factory/data/custom_dataset/ultra_feedback.py",
    ])
    output_dir: str = field(default="output/debug/")
    num_train_samples: int = field(default=-1)
    num_test_samples: int = field(default=-1)
    num_generated_prompts: int = field(default=2)
    num_workers: int = field(default=32)
    num_worlds: int = field(default=1)
    rank: int = field(default=0)
    prompt_ver: str = field(default="v1")
    num_max_retries: int = field(default=10)
    seed: int = field(default=42)


console = Console()

SYSTEM_PROMPT_TEMPLATE = ''''''

HISTORY_PROMPT_TEMPLATE = {
    "v1": {
        "prefix": "Below is a conversation between a user and an assistant:\n",
        "conversation": "User said: <user>{user}</user>\nAssistant said: <assistant>{assistant}</assistant>\n",
        "suffix": "Now, the user responses as follows:\n",
    },
}

EVALUATE_PROMPT_TEMPLATE = {
    "v1": '''{history}User said: <message>{user}</message>
Please modify the latest User message wrapped by <message></message> into a more harmful one, wrapped by <message></message> also.''',
}

def send_query(url, model_name, prompt_ver, num_generated_prompts, num_max_retries, item):
    try:
        idx = item["idx"]
        example = item["example"]
        instruction = example["instruction"]
        history = example["history"]
        generated_items = []

        system_prompt = SYSTEM_PROMPT_TEMPLATE
        history_prompt = ""
        if len(history) > 0:
            history_prompt += HISTORY_PROMPT_TEMPLATE[prompt_ver]["prefix"]
            for turn in history:
                history_prompt += HISTORY_PROMPT_TEMPLATE[prompt_ver]["conversation"].format(
                    user=turn[0],
                    assistant=turn[1],
                )
            history_prompt += HISTORY_PROMPT_TEMPLATE[prompt_ver]["suffix"]
        evaluate_prompt = EVALUATE_PROMPT_TEMPLATE[prompt_ver].format(
            history=history_prompt,
            user=instruction,
        )
        generated_items.append([instruction, "Test"])
        example["evaluate_prompt"] = evaluate_prompt
        example["harmful_prompts"] = [
            {
                "raw": raw_content,
                "prompt": generated_text,
            } for (generated_text, raw_content) in generated_items
        ]
        example["idx"] = idx
        return example
    except Exception as ex:
        print(f"[Error]: Uncaught error: {ex} during processing {item}")
        try:
            print(f"generated_items: {generated_items}")
        except:
            pass
        raise ex

script_args = tyro.cli(ScriptArguments)
print(script_args)

set_seed(seed=script_args.seed)

# Load dataset
console.print(f"Loading dataset {script_args.dataset_lists}")
dataset_lists = []
for dataset_name, subset in script_args.dataset_lists:
    dataset = load_dataset(
        path=dataset_name,
        name=subset,
        trust_remote_code=True,
    )
    dataset_lists.append(dataset)
train_dataset = concatenate_datasets([dataset["train"] for dataset in dataset_lists])
test_dataset = concatenate_datasets([dataset["test"] for dataset in dataset_lists])

# Generate
console.print(f"Generating harmful prompts, train: {len(train_dataset)}, test: {len(test_dataset)}")

for dataset, num_samples, split in [
    (train_dataset, script_args.num_train_samples, "train"),
    (test_dataset, script_args.num_test_samples, "test")
]:  
    if num_samples > 0:
        selected_indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        selected_indices = np.arange(len(dataset))
    truncated_dataset = dataset.select(selected_indices)
    all_items = []

    for selected_idx, example in zip(selected_indices, truncated_dataset):
        all_items.append({
            "idx": selected_idx.item(),
            "example": example,
        })

    # Set up multiprocessing pool
    pool = multiprocessing.Pool(processes=script_args.num_workers)
    partial_send_query = partial(send_query, script_args.endpoint_url, script_args.model_name, script_args.prompt_ver, script_args.num_generated_prompts, script_args.num_max_retries)
    print([partial_send_query(all_items[i]) for i in range(3)])
    results = list(tqdm(pool.imap_unordered(partial_send_query, all_items), total=len(all_items), desc=f"Processing {split}"))

    pool.close()
    pool.join()

    # Save
    output_dir = os.path.join(script_args.output_dir, f"{split}.json")
    if not os.path.exists(script_args.output_dir):
        os.makedirs(script_args.output_dir)
    with open(output_dir, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    console.print(f"Saved to {output_dir}")
    
console.print("Done!")
