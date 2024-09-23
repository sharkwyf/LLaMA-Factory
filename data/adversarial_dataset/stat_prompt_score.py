import tyro
import numpy as np
import json
import multiprocessing
import requests
import os
from tqdm import tqdm
from rich.prompt import Prompt
from rich.console import Console
from rich.table import Table
from rich import print
from dataclasses import dataclass, field
from tqdm import tqdm
from datetime import datetime
from functools import partial
from datasets import load_dataset, concatenate_datasets
from typing import List

prefix_path = "data/adversarial_dataset/exp"
max_iters = 30

@dataclass(kw_only=True)
class ScriptArguments:
    data_path: List[str] = field(default_factory=lambda: [
        # attacker
        f"{prefix_path}/eval/score-base",
        *[
            f"{prefix_path}/eval/score-iterate-{i}"
            for i in range(1, 1 + max_iters)
        ],

        # attacker vs previous defender
        "section_line",
        *[
            f"{prefix_path}/eval/score-iterate-{i}-prev"
            for i in range(1, 1 + max_iters)
        ],
    ])
    seed: int = field(default=42)


console = Console()

script_args = tyro.cli(ScriptArguments)
print(script_args)

# set_seed(seed=script_args.seed)

table = Table(title="Harmful Prompts Statistics", title_style="bold magenta")

# Define tables columns
table.add_column("Harmful Prompt", style="dim")
table.add_column("Total", style="dim")
table.add_column("Skipped", style="dim")
table.add_column("Base prompt reward", style="dim")
table.add_column("Average harmful reward", style="dim")
table.add_column("First harmful reward", style="dim")
table.add_column("Last harmful reward", style="dim")

# open the data file
for path in script_args.data_path:
    if path == "section_line":
        table.add_section()
        continue
    
    for split in ["test"]:
        if not os.path.exists(os.path.join(path, f"{split}.json")):
            print(f"File {os.path.join(path, f'{split}.json')} not found")
            continue
        with open(os.path.join(path, f"{split}.json"), "r") as f:
            data = json.load(f)

        base_rewards = []
        first_rewards = []
        last_rewards = []
        all_harmful_rewards = []
        all_cnt = 0
        skipped_cnt = 0
        for example in data:
            if not "chosen_reward" in example or not "rejected_reward" in example:
                skipped_cnt += 1
                print(f'Skipped example: {example}')
                continue
            base_rewards.append([example["chosen_reward"], example["rejected_reward"]])

            generated_items = [item for item in example["generated_prompts"] if "chosen_reward" in item and "rejected_reward" in item]
            if len(generated_items) < len(example["generated_prompts"]):
                skipped_cnt += len(example["generated_prompts"]) -  len(generated_items)
                print(f'Skipped harmful prompt: {[item for item in example["generated_prompts"] if not ("chosen_reward" in item and "rejected_reward" in item)]}')
            generated_items = sorted(generated_items, key=lambda x: x["chosen_reward"] - x["rejected_reward"])
            first_rewards.append([generated_items[0]["chosen_reward"], generated_items[0]["rejected_reward"]])
            last_rewards.append([generated_items[-1]["chosen_reward"], generated_items[-1]["rejected_reward"]])

            for harmful_item in generated_items:
                chosen, rejected = harmful_item["chosen_reward"], harmful_item["rejected_reward"]
                all_harmful_rewards.append([chosen, rejected])
                all_cnt += 1

        chosen_rewards, rejected_rewards = np.array(base_rewards).T
        first_chosen_rewards, first_rejected_rewards = np.array(first_rewards).T
        last_chosen_rewards, last_rejected_rewards = np.array(last_rewards).T
        all_chosen_rewards, all_rejected_rewards = np.array(all_harmful_rewards).T

        # all_rewards = np.array(all_rewards)

        # chosen_rewards, rejected_rewards, first_chosen_rewards, first_rejected_rewards, second_chosen_rewards, second_rejected_rewards = all_rewards.T

        base_rewards = (chosen_rewards - rejected_rewards)
        first_harmful_rewards = (first_chosen_rewards - first_rejected_rewards)
        last_harmful_rewards = (last_chosen_rewards - last_rejected_rewards)
        all_harmful_rewards = (all_chosen_rewards - all_rejected_rewards)

        table.add_row(
            f"{'/'.join(path.split('/')[-2:])}:{split}",
            f"{all_cnt}",
            f"{skipped_cnt}",
            f"{base_rewards.mean():.2f} ± {base_rewards.std():.2f} ({chosen_rewards.mean():.2f} - {rejected_rewards.mean():.2f})",
            f"{all_harmful_rewards.mean():.2f} ± {all_harmful_rewards.std():.2f} ({all_chosen_rewards.mean():.2f} - {all_rejected_rewards.mean():.2f})",
            f"{first_harmful_rewards.mean():.2f} ± {first_harmful_rewards.std():.2f} ({first_chosen_rewards.mean():.2f} - {first_rejected_rewards.mean():.2f})",
            f"{last_harmful_rewards.mean():.2f} ± {last_harmful_rewards.std():.2f} ({last_chosen_rewards.mean():.2f} - {last_rejected_rewards.mean():.2f})",
        )

        print(f"Skipped {skipped_cnt} examples")

print(table)
print("Done")