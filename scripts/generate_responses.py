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
from typing import Awaitable, List, Tuple, Optional, Literal

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
    raw_data_path: str = field(default="data/adversarial_dataset/exp2/iterate-1/prompt")
    dataset_splits: tuple[Literal["train", "test"], ...] = field(default=("train", "test"))
    num_prompts_per_example: int = field(default=2)
    num_generated_responses: int = field(default=2)
    prompt_ver: str = field(default="v1")
    output_dir: str = field(default="output/debug/")

@dataclass
class VLLMEngineArgs(AsyncEngineArgs):
    model: str = field(default="output/dpo/mistral_7b-adversarial-raw-sigmoid")
    disable_custom_all_reduce: bool = True


console = Console()

async def run_request(chat_serving: OpenAIServingChat,
                      request: BatchRequestInput,
                      pbar) -> BatchRequestOutput:
    chat_request = request.body
    chat_response = await chat_serving.create_chat_completion(chat_request)
    if isinstance(chat_response, ChatCompletionResponse):
        batch_output = BatchRequestOutput(
            id=f"vllm-{random_uuid()}",
            custom_id=request.custom_id,
            response=chat_response,
            error=None,
        )
    else:
        batch_output = BatchRequestOutput(
            id=f"vllm-{random_uuid()}",
            custom_id=request.custom_id,
            response=None,
            error=chat_response,
        )
    pbar.update(1)
    return batch_output

async def main(args: ScriptArguments, engine_args: VLLMEngineArgs):
    set_seed(seed=engine_args.seed)

    served_model_names = [engine_args.model]

    engine_args = AsyncEngineArgs.from_cli_args(engine_args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # When using single vLLM without engine_use_ray
    model_config = await engine.get_model_config()

    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names,
        "assistant",
    )
    
    for split in args.dataset_splits:
        # Load dataset
        console.print(f"Loading raw data {split} from {args.raw_data_path}")
        with open(os.path.join(args.raw_data_path, f"{split}.json"), "r") as f:
            data = json.load(f)
            
        # Submit all requests in the file to the engine "concurrently".
        response_futures: List[Awaitable[BatchRequestOutput]] = []
        pbar = tqdm_async(total=len(data))
        for i, example in enumerate(data):
            instruction = example["instruction"]
            history = example["history"]
            generated_items = example["generated_prompts"]

            history_messages = []
            for user, assistant in history:
                history_messages.extend([
                    {
                        "role": "user",
                        "content": user,
                    },
                    {
                        "role": "assistant",
                        "content": assistant,
                    }
                ])

            for j, harmful_item in enumerate(generated_items[:args.num_prompts_per_example]):
                messages = history_messages + [
                    {
                        "role": "user",
                        "content": harmful_item["prompt"],
                    }
                ]

                request = BatchRequestInput(
                    custom_id = f"{i}-{j}",
                    method = "POST",
                    url = "/v1/chat/completions",
                    body = ChatCompletionRequest(
                        **{
                            "messages": messages,
                            "model": engine_args.model,
                            "n": args.num_generated_responses,
                            "stream": False,
                            "max_tokens": 1024,
                            "temperature": 1.0,
                            "top_p": 0.9,
                            "frequency_penalty": 1.1,
                            "stop": ["</s>", "<|im_end|>", "</message>"],
                            # "stop_token_ids": [128009],
                            "include_stop_str_in_output": True,
                            # "skip_special_tokens": False,
                        }
                    )
                )

                response_futures.append(run_request(openai_serving_chat, request, pbar))
        
        pbar.total = len(response_futures)
        chunk_size = 1024
        print(f"Start processing in batch size: {chunk_size}")
        responses = []
        for i in range(0, len(response_futures), chunk_size):
            chunk = response_futures[i:i+chunk_size]
            responses.extend(await asyncio.gather(*chunk))
        
        pattern = r"^(\d+)-(\d+)$"
        for response in responses:
            match = re.match(pattern, response.custom_id)
            if match:
                i, j = map(int, match.groups())
                data[i]["generated_prompts"][j]["generated_responses"] = [
                    {
                        "response": choice.message.content,
                    }
                    for choice in response.response.choices
                ]
            else:
                print(f"Invalid custom_id format: {response.custom_id}")

        # Save
        output_dir = os.path.join(args.output_dir, f"{split}.json")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(output_dir, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        console.print(f"Saving generated harmful prompts {split} to {output_dir}")

    console.print("Done!")
    exit()


if __name__ == "__main__":
    scripts_args, engine_args = tyro.cli(
        tyro.conf.OmitArgPrefixes[Tuple[ScriptArguments, VLLMEngineArgs]]
    )
    console.log(f"Script arguments: {scripts_args}\nEngine arguments: {engine_args}")
    asyncio.run(main(scripts_args, engine_args), debug=False)
