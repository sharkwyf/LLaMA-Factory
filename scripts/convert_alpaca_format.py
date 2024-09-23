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
    raw_data_path: str = field(default="data/adversarial_dataset/exp/base/prompt/test.json")
    dataset_name: str = field(default="hh_rlhf_harmless")
    test_samples_per_iter: int = field(default=-1)
    output_dir: str = field(default="data/adversarial_dataset/exp/eval/")
    as_base: bool = field(default=False)
    seed: int = field(default=42)


console = Console()

async def main(script_args: ScriptArguments):
    set_seed(seed=script_args.seed)

    # Load data
    with open(script_args.raw_data_path, "r") as f:
        all_items = json.load(f)

    # extract dirname and "base" from dirname
    dirname = os.path.dirname(script_args.raw_data_path)
    if script_args.as_base:
        subfolder = "base"
    else:
        subfolder = dirname.split("/")[-2]

    results = []
    if script_args.test_samples_per_iter > 0:
        all_items = all_items[:script_args.test_samples_per_iter]
    for example in all_items:
        results.append({
            "instruction": example["instruction"],
            "output": example["instruction"] if script_args.as_base else example["generated_prompts"][0]["prompt"],
            "generator": f"{script_args.dataset_name}-prompt-{subfolder}",
            "dataset": script_args.dataset_name,
        })

    # Save
    output_dir = script_args.output_dir
    save_path = os.path.join(output_dir, f"{script_args.dataset_name}-prompt-{subfolder}.json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    console.print(f"Saving alpaca output to {save_path}")
        
    console.print("Done!")
    exit()


if __name__ == "__main__":
    scripts_args = tyro.cli(ScriptArguments)
    console.log(f"Script arguments: {scripts_args}")
    asyncio.run(main(scripts_args))
