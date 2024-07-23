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
    endpoint_url: str = field(default="http://10.140.0.159:33998/v1")
    model_name: str = field(default="mistral", metadata={"help": '["Qwen1.5-72B-Chat", "Llama3-70B", "Llama3-70B-Instruct"]'})
    raw_data_path: str = field(default="data/adversarial_dataset/iterate-0/harmful_prompt/")
    output_dir: str = field(default="data/adversarial_dataset/iterate-0/harmful_response/")
    num_workers: int = field(default=32)
    num_prompts_per_example: int = field(default=16)
    num_generated_responses: int = field(default=1)
    num_max_retries: int = field(default=10)
    seed: int = field(default=42)


console = Console()

def send_query(url, model_name, num_prompts_per_example, num_generated_responses, num_max_retries, item):
    try:
        try:
            idx = item["idx"]
            example = item["example"]
            instruction = example["instruction"]
            history = example["history"]

            for harmful_item in example["harmful_prompts"][:num_prompts_per_example]:
                generated_items = []
                generated_cnt = 0
                retry_cnt = 0
                messages = []

                # messages.extend([
                #     {
                #         "role": "system",
                #         "content": "",
                #     },
                # ])
                messages.extend([
                    {
                        "role": role,
                        "content": content,
                    } for turn in history for role, content in zip(["user", "assistant"], turn)
                ])
                messages.extend([
                    {
                        "role": "user",
                        "content": harmful_item["prompt"],
                    },
                ])
                
                data = {
                    "model": model_name,
                    "messages": messages,
                    "n": 1,
                    "stream": False,
                    "max_tokens": 2048,
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "frequency_penalty": 1.1,
                    "stop": ["</s>", "<|im_end|>"],
                    # "stop_token_ids": [128009],
                    # "include_stop_str_in_output": True,
                    # "skip_special_tokens": False,
                }
                while generated_cnt < num_generated_responses:
                    if retry_cnt == num_max_retries:
                        print(f"[Error]: Retried {retry_cnt} times, skip this example: {example}\ngenerated: {generated_items}\n")
                        break
                    response = requests.post(
                        url=f"{url}/chat/completions",
                        json=data,
                    )
                    raw_content = response.json()["choices"][0]["message"]["content"]
                    generated_text = raw_content.strip()
                    # print(f"\nGenerated Output: {generated_text}\n")
                    retry_cnt += 1
                    if generated_text not in [x[0] for x in generated_items]:
                        generated_items.append([generated_text, raw_content])
                        generated_cnt += 1
                        retry_cnt = 0
                harmful_item["generated_responses"] = [{
                    "response": generated_text,
                } for generated_text, raw_content in generated_items]
        except Exception as ex:
            print(f"[Error]: {ex} during processing '{generated_text}'")
            try:
                print(f"[response]: {response}")
                print(f"[Error]: HTTP {response.status_code} - {response.reason}")
            except:
                pass
            generated_items = f"[Error]: HTTP {response.status_code} - {response.reason} during process '{generated_text}'"
            raise ex
        finally:
            return idx, example
    except Exception as ex:
        print(f"[Error]: Uncaught error: {ex} during processing {item}")
        raise ex

script_args = tyro.cli(ScriptArguments)
print(script_args)

set_seed(seed=script_args.seed)

for split in ["train", "test"]:
    # Load dataset
    with open(os.path.join(script_args.raw_data_path, f"{split}.json"), "r") as f:
        data = json.load(f)

    all_items = [{"idx": idx, "example": example} for idx, example in enumerate(data)]

    # Generate
    console.print(f'Generating harmful responses, {script_args.raw_data_path}/{split}: {len(all_items)}')

    # Set up multiprocessing pool
    pool = multiprocessing.Pool(processes=script_args.num_workers)
    partial_send_query = partial(send_query, script_args.endpoint_url, script_args.model_name, script_args.num_prompts_per_example, script_args.num_generated_responses, script_args.num_max_retries)
    print([partial_send_query(all_items[i]) for i in range(3)])
    results = list(tqdm(pool.imap_unordered(partial_send_query, all_items), total=len(all_items), desc=f"Processing {split}"))
    results = sorted(results, key=lambda x: x[0])
    results = [example for idx, example in results]

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
