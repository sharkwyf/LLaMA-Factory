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
from transformers import set_seed
from tqdm import tqdm
from datetime import datetime
from functools import partial
from datasets import load_dataset, concatenate_datasets
from typing import List

prefix_path = "data/adversarial_dataset/exp"

@dataclass(kw_only=True)
class ScriptArguments:
    data_path: List[str] = field(default_factory=lambda: [
        # attacker
        f"{prefix_path}/eval/prompt-base",
        f"{prefix_path}/eval/prompt-iterate-1",
        f"{prefix_path}/eval/prompt-iterate-2",
        f"{prefix_path}/eval/prompt-iterate-3",
        f"{prefix_path}/eval/prompt-iterate-4",
        
        "section_line",

        # baseline
        f"{prefix_path}/eval/response-mistral",
        f"{prefix_path}/eval/response-base-baseline",
        f"{prefix_path}/eval/response-adversarial-1-baseline",
        f"{prefix_path}/eval/response-adversarial-2-baseline",
        f"{prefix_path}/eval/response-adversarial-3-baseline",
        f"{prefix_path}/eval/response-adversarial-4-baseline",
        
        "section_line",

        # defender
        f"{prefix_path}/eval/response-mistral",
        f"{prefix_path}/eval/response-base-defender",
        f"{prefix_path}/eval/response-adversarial-1-defender",
        f"{prefix_path}/eval/response-adversarial-2-defender",
        f"{prefix_path}/eval/response-adversarial-3-defender",
        f"{prefix_path}/eval/response-adversarial-4-defender",
        
        "section_line",
        
        # defender-mixed
        f"{prefix_path}/eval/response-mistral",
        f"{prefix_path}/eval/response-base-defender",
        f"{prefix_path}/eval/response-adversarial-1-defender-mixed",
        f"{prefix_path}/eval/response-adversarial-2-defender-mixed",
        f"{prefix_path}/eval/response-adversarial-3-defender-mixed",
        f"{prefix_path}/eval/response-adversarial-4-defender-mixed",
    ])
    seed: int = field(default=42)


console = Console()

script_args = tyro.cli(ScriptArguments)
print(script_args)

set_seed(seed=script_args.seed)

table = Table(title="Harmful Prompts Statistics", title_style="bold magenta")

# Define tables columns
table.add_column("Iteration / Generated Response", style="dim")
table.add_column("Total", style="dim")
table.add_column("Skipped", style="dim")
table.add_column("RM Score", style="dim")

# open the data file
for path in script_args.data_path:
    if path == "section_line":
        table.add_section()
        continue
    # for split in ["train", "test"]:
    for split in ["test"]:
        if not os.path.exists(os.path.join(path, f"{split}.json")):
            print(f"File {os.path.join(path, f'{split}.json')} not found")
            continue
        with open(os.path.join(path, f"{split}.json"), "r") as f:
            data = json.load(f)

        all_scores = []
        skipped_cnt = 0

        for example in data:
            for harmful_item in example["harmful_prompts"]:
                if "generated_responses" not in harmful_item:
                    continue
                for response_item in harmful_item["generated_responses"]:
                    if "score" not in response_item:
                        continue
                    score = response_item["score"]
                    all_scores.append(score)
        all_scores = np.array(all_scores)

        table.add_row(
            f"{'/'.join(path.split('/')[-2:])}:{split}",
            f"{len(all_scores)}",
            f"{skipped_cnt}",
            f"{all_scores.mean():.2f} ± {all_scores.std():.2f}",
        )

print(f"Skipped {skipped_cnt} examples")
print(table)

print("Done")