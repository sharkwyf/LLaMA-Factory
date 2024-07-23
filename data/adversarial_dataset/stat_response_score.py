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

@dataclass(kw_only=True)
class ScriptArguments:
    data_path: List[str] = field(default_factory=lambda: [
        # attacker
        "data/adversarial_dataset/exp/eval/prompt-base",
        "data/adversarial_dataset/exp/eval/prompt-iterate-1",
        "data/adversarial_dataset/exp/eval/prompt-iterate-2",
        "data/adversarial_dataset/exp/eval/prompt-iterate-3",
        "data/adversarial_dataset/exp/eval/prompt-iterate-4",
        
        "section_line",

        # baseline
        "data/adversarial_dataset/exp/eval/response-mistral",
        "data/adversarial_dataset/exp/eval/response-base-baseline",
        "data/adversarial_dataset/exp/eval/response-adversarial-1-baseline",
        "data/adversarial_dataset/exp/eval/response-adversarial-2-baseline",
        "data/adversarial_dataset/exp/eval/response-adversarial-3-baseline",
        "data/adversarial_dataset/exp/eval/response-adversarial-4-baseline",
        
        "section_line",

        # defender
        "data/adversarial_dataset/exp/eval/response-mistral",
        "data/adversarial_dataset/exp/eval/response-base-defender",
        "data/adversarial_dataset/exp/eval/response-adversarial-1-defender",
        "data/adversarial_dataset/exp/eval/response-adversarial-2-defender",
        "data/adversarial_dataset/exp/eval/response-adversarial-3-defender",
        "data/adversarial_dataset/exp/eval/response-adversarial-4-defender",

        # # old
        # "section_line",

        # # attacker
        # "data/adversarial_dataset/iterate-0/eval-response-mistral_7b-adversarial-raw-sigmoid",
        # "data/adversarial_dataset/iterate-1/eval-response-mistral_7b-adversarial-raw-sigmoid",
        # "data/adversarial_dataset/iterate-2/eval-response-mistral_7b-adversarial-raw-sigmoid",
        # "data/adversarial_dataset/iterate-3/eval-response-mistral_7b-adversarial-raw-sigmoid",

        # "section_line",

        # # defender
        # "data/adversarial_dataset/iterate-0/eval-response-mistral_7b",
        
        # "data/adversarial_dataset/iterate-0/eval-response-mistral_7b-adversarial-raw-sigmoid",
        # "data/adversarial_dataset/iterate-0/eval-response-mistral_7b-adversarial-raw-sigmoid-adversarial-raw-sigmoid",
        # "data/adversarial_dataset/iterate-0/eval-response-mistral_7b-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid",
        # "data/adversarial_dataset/iterate-0/eval-response-mistral_7b-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid",
        # "data/adversarial_dataset/iterate-0/eval-response-mistral_7b-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid",

        # "section_line",
        
        # "data/adversarial_dataset/iterate-0/eval-response-mistral_7b-adversarial-0-defender-sigmoid",

        # "data/adversarial_dataset/iterate-0/eval-response-mistral_7b-adversarial-raw-sigmoid-adversarial-0-defender-sigmoid",
        # "data/adversarial_dataset/iterate-0/eval-response-mistral_7b-adversarial-raw-sigmoid-adversarial-0-defender-sigmoid-adversarial-1-defender-sigmoid",
        # "data/adversarial_dataset/iterate-0/eval-response-mistral_7b-adversarial-raw-sigmoid-adversarial-0-defender-sigmoid-adversarial-1-defender-sigmoid-adversarial-2-defender-sigmoid",
        # "data/adversarial_dataset/iterate-0/eval-response-mistral_7b-adversarial-raw-sigmoid-adversarial-0-defender-sigmoid-adversarial-1-defender-sigmoid-adversarial-2-defender-sigmoid-adversarial-3-defender-sigmoid",
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