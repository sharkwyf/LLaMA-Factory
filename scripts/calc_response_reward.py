# coding=utf-8
# Calculates the ppl on the dataset of the pre-trained models.
# Usage: python cal_ppl.py --model_name_or_path path_to_model --save_name ppl.json

import sys; sys.path.append('./src/')

import argparse
import os
import json
from multiprocessing import Pool
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence

import tyro
import torch
import torch.nn.functional as F
from datasets import Dataset
from rich import print
from accelerate import Accelerator, DistributedType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, DataCollatorWithPadding

from llamafactory.data import get_dataset
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_custom_optimzer, create_custom_scheduler, get_batch_logps
from llamafactory.data.data_utils import Role
from llamafactory.data.processors.supervised import _encode_supervised_example
from llamafactory.data.template import TEMPLATES


@dataclass(kw_only=True)
class ScriptArguments:
    stage: str = field(default="sft")
    model_name_or_path: str = field(default="output/rm/mistral_7b-adversarial-raw-1")
    raw_data_path: str = field(default="data/adversarial_dataset/iterate-0/harmful_response/")
    dataset_splits: tuple[Literal["train", "test"], ...] = field(default=("train", "test"))
    save_name: str = field(default="data/adversarial_dataset/iterate-0/harmful_response_scored/")
    batch_size: int = field(default=8)
    preprocessing_num_workers: Optional[int] = field(default=1)
    dataloader_num_workers: Optional[int] = field(default=8)
    template: Literal["default", "mistral"] = field(default="mistral")
    cutoff_len: int = field(default=2048)
    bf16: bool = field(default=False)
    flash_attn: Literal["off", "sdpa", "fa2", "auto"] = field(default="off")

@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    cache = []

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        indices = [feature.pop("index") for feature in features]

        batch = super().__call__(features)

        return batch, indices


def main(args: ScriptArguments):
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        dict(
            stage=args.stage,
            model_name_or_path=args.model_name_or_path,
            dataset="alpaca_en_demo",
            preprocessing_num_workers=args.preprocessing_num_workers,
            dataloader_num_workers=args.dataloader_num_workers,
            dataset_dir="data",
            template=args.template,
            cutoff_len=args.cutoff_len,
            output_dir="dummy_dir",
            bf16=args.bf16,
            flash_attn=args.flash_attn,
        )
    )

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print("Arguments: ", args)

    device = accelerator.device

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=True)

    for split in args.dataset_splits:
        with open(os.path.join(args.raw_data_path, f"{split}.json"), "r") as f:
            raw_data = json.load(f)
            
        def gen():
            template = TEMPLATES.get(data_args.template, None)
            for i, item in enumerate(raw_data):
                history = []
                for old_prompt, old_response in item["history"]:
                    history.append({"role": Role.USER.value, "content": old_prompt})
                    history.append({"role": Role.ASSISTANT.value, "content": old_response})
                for j, prompt_item in enumerate(item["harmful_prompts"]):
                    if "generated_responses" not in prompt_item:
                        continue
                    for k, response_item in enumerate(prompt_item["generated_responses"]):
                        prompt = history + [{"role": Role.USER.value, "content": prompt_item["prompt"]}]
                        response = [{"role": Role.ASSISTANT.value, "content": response_item["response"]}]
                        system = ""
                        tools = ""

                        input_ids, labels = _encode_supervised_example(
                            prompt=prompt,
                            response=response,
                            system=system,
                            tools=tools,
                            template=template,
                            tokenizer=tokenizer,
                            processor=None,
                            data_args=data_args,
                        )
                                    
                        yield {
                            "index": (i, j, k),
                            "input_ids": input_ids,
                            "labels": labels,
                        }

        dataset = Dataset.from_generator(gen, num_proc=data_args.preprocessing_num_workers)
        print(f"Found {len(dataset)} responses to calculate rewards in {args.raw_data_path}/{split}.json")

        data_collator = PairwiseDataCollatorWithPadding(
            tokenizer=tokenizer, pad_to_multiple_of=8
        )
        
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator, num_workers=args.dataloader_num_workers, pin_memory=True)

        model, dataloader = accelerator.prepare(model, dataloader)

        change_list = []
        with torch.no_grad():
            for inputs, indices in tqdm(dataloader, desc="Calculating response rewards"):
                inputs = {k: v.to(device) for k, v in inputs.items()}

                _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)

                batch_size = inputs["input_ids"].size(0)
                chosen_input_ids = inputs["input_ids"]
                chosen_rewards = values
                chosen_scores = []

                for i in range(batch_size):
                    chosen_length = (chosen_input_ids[i] != tokenizer.pad_token_id).nonzero()[-1] + 1
                    chosen_end_rewards = chosen_rewards[i, chosen_length - 1]
                    chosen_scores.append(chosen_end_rewards.cpu().item())

                for index, score in zip(indices, chosen_scores):
                    change_list.append([index, score])
                    
        print(f"Rank {accelerator.local_process_index}: Processed {len(change_list)} items")
                    
        accelerator.wait_for_everyone()

        # Gather data across all devices, use gather_object if non-tensor data is involved
        gathered_change_list = accelerator.gather_for_metrics(change_list, use_gather_object=True)
                    
        if accelerator.is_main_process:
            for index, score in tqdm(gathered_change_list, desc="Processing gathered inputs"):
                i, j, k = index
                raw_data[i]["harmful_prompts"][j]["generated_responses"][k]["score"] = score
            
            save_path = os.path.join(args.save_name, f"{split}.json")
            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=4, ensure_ascii=False)
            print("Labeled dataset have been saved at {}.".format(save_path))


if __name__ == "__main__":
    args = tyro.cli(ScriptArguments)
    main(args)