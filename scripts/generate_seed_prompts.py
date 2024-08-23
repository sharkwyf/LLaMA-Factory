import tyro
import numpy as np
import json
import multiprocessing
import requests
import os
import asyncio
import re
from tqdm import tqdm
from rich import print
from rich.prompt import Prompt
from rich.console import Console
from dataclasses import dataclass, field
from transformers import set_seed
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async
from datetime import datetime
from functools import partial
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

import asyncio
from io import StringIO
from typing import Awaitable, List, Tuple, Optional

import aiohttp

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs, nullable_str
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (BatchRequestInput,
                                              BatchRequestOutput,
                                              ChatCompletionRequest,
                                              ChatCompletionResponse)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid

@dataclass(kw_only=True)
class ScriptArguments:
    dataset_lists: tuple[tuple[str, str], ...] = field(default_factory=lambda: [
        # "/mnt/petrelfs/wangyuanfu/workspace/LLaMA-Factory/data/custom_dataset/helpsteer.py",
        ["/mnt/petrelfs/wangyuanfu/workspace/LLaMA-Factory/data/custom_dataset/hh_rlhf.py", "harmless"]
        # "/mnt/petrelfs/wangyuanfu/workspace/LLaMA-Factory/data/custom_dataset/safe_rlhf.py",
        # "/mnt/petrelfs/wangyuanfu/workspace/LLaMA-Factory/data/custom_dataset/ultra_feedback.py",
    ])
    num_train_samples: int = field(default=-1)
    num_test_samples: int = field(default=-1)
    output_dir: str = field(default="output/debug/")
    seed: int = field(default=42)


console = Console()


async def run_request(item, pbar):
    idx, example = item["idx"], item["example"]
    example["harmful_prompts"] = [
        {
            "raw": "Test",
            "prompt": example["instruction"],
        }
    ]
    pbar.update(1)
    return example

async def main(script_args: ScriptArguments):
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
    console.print(f"Generating seed prompts, train: {len(train_dataset)}, test: {len(test_dataset)}")

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
            
        # Submit all requests in the file to the engine "concurrently".
        response_futures: List[Awaitable[BatchRequestOutput]] = []
        pbar = tqdm_async(total=len(all_items))
        for item in all_items:
            response_futures.append(run_request(item, pbar))

        chunk_size = 1000
        print(f"Start processing in batch size: {chunk_size}")
        results = []
        for i in range(0, len(response_futures), chunk_size):
            chunk = response_futures[i:i+chunk_size]
            results.extend(await asyncio.gather(*chunk))

        # Save
        output_dir = os.path.join(script_args.output_dir, f"{split}.json")
        if not os.path.exists(script_args.output_dir):
            os.makedirs(script_args.output_dir)
        with open(output_dir, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        console.print(f"Saving seed prompts {split} to {output_dir}")
        
    console.print("Done!")
    exit()


if __name__ == "__main__":
    scripts_args = tyro.cli(ScriptArguments)
    console.log(f"Script arguments: {scripts_args}")
    asyncio.run(main(scripts_args))
