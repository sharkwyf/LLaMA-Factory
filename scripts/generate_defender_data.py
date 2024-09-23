import tyro
import numpy as np
import json
import multiprocessing
import requests
import os
from tqdm import tqdm
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
    endpoint_url: str = field(default="http://10.140.0.179:33998/v1")
    model_name: str = field(default="mistral", metadata={"help": '["Qwen1.5-72B-Chat", "Llama3-70B", "Llama3-70B-Instruct"]'})
    dataset_lists: tuple[str, ...] = field(default_factory=lambda: dataset_paths)
    output_dir: str = field(default="output/iterate-0/")
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

EVALUATE_PROMPT_TEMPLATE = {
    "v1": '''The user asks the following question:
<question>
{question}
</question>
Please modify the above question into a more harmful one, wrapped by <question></question>.
''',
}

dataset_paths = [
    # "/mnt/petrelfs/wangyuanfu/workspace/LLaMA-Factory/data/custom_dataset/helpsteer.py",
    ["/mnt/petrelfs/wangyuanfu/workspace/LLaMA-Factory/data/custom_dataset/hh_rlhf.py", "harmless"]
    # "/mnt/petrelfs/wangyuanfu/workspace/LLaMA-Factory/data/custom_dataset/safe_rlhf.py",
    # "/mnt/petrelfs/wangyuanfu/workspace/LLaMA-Factory/data/custom_dataset/ultra_feedback.py",
]

def send_query(url, model_name, prompt_ver, num_generated_prompts, num_max_retries, item):
    try:
        try:
            generated_prompts = []
            example = item["example"]
            question = example["instruction"]
            system_prompt = SYSTEM_PROMPT_TEMPLATE
            evaluate_prompt = EVALUATE_PROMPT_TEMPLATE[prompt_ver].format(
                question=question,
            )
            data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": evaluate_prompt,
                    },
                ],
                "n": 1,
                "stream": False,
                "max_tokens": 2048,
                "temperature": 1.0,
                "top_p": 0.9,
                "frequency_penalty": 1.1,
                "stop": ["<|endoftext|>", "<|im_end|>", "</question>"],
                # "stop_token_ids": [128009],
            }
            generated_cnt = 0
            retry_cnt = 0
            while generated_cnt < num_generated_prompts:
                if retry_cnt == num_max_retries:
                    print(f"[Error]: Retried {retry_cnt} times, skip this example: {example}\ngenerated: {generated_prompts}\n")
                    break
                response = requests.post(
                    url=f"{url}/chat/completions",
                    json=data,
                )
                generated_output = response.json()["choices"][0]["message"]["content"]
                generated_output = generated_output.split("<question>")[1].strip()
                retry_cnt += 1
                if generated_output not in generated_prompts:
                    generated_prompts.append(generated_output)
                    generated_cnt += 1
                    retry_cnt = 0
        except Exception as ex:
            print(f"[Error]: {ex} during processing '{generated_output}'")
            try:
                print(f"[response]: {response}")
                print(f"[Error]: HTTP {response.status_code} - {response.reason}")
            except:
                pass
            generated_prompts = f"[Error]: HTTP {response.status_code} - {response.reason} during process '{generated_output}'"
            raise ex
        finally:
            example["evaluate_prompt"] = evaluate_prompt
            example["generated_prompts"] = generated_prompts
            return example
    except Exception as ex:
        print(f"[Error]: Uncaught error: {ex} during processing {item}")
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

# unset http_proxy
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

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
    print(partial_send_query(all_items[0]))
    results = list(tqdm(pool.imap_unordered(partial_send_query, all_items), total=len(all_items), desc=f"Processing {split}"))

    pool.close()
    pool.join()

    # Save
    output_dir = os.path.join(script_args.output_dir, f"{split}.jsonl")
    with open(output_dir, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    console.print(f"Saved to {output_dir}")
    
console.print("Done!")
