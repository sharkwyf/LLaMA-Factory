# coding=utf-8
# Calculates the ppl on the dataset of the pre-trained models.
# Usage: python cal_ppl.py --model_name_or_path path_to_model --save_name ppl.json

import sys; sys.path.append('./src/')

import os
import json
from multiprocessing import Pool
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence

import tyro
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, DataCollatorWithPadding
from accelerate import Accelerator, DistributedType
from tqdm import tqdm
from rich import print

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_custom_optimzer, create_custom_scheduler, get_batch_logps, create_ref_model
from llamafactory.data.data_utils import Role
from llamafactory.data.processors.pairwise import _encode_pairwise_example
from llamafactory.data.template import TEMPLATES


@dataclass(kw_only=True)
class ScriptArguments:
    stage: str = field(default="rm")
    model_name_or_path: str = field(default="facebook/opt-350m")
    raw_data_path: str = field(default="data/adversarial_dataset/iterate-0/prompt-mistral_7b/")
    dataset_splits: tuple[Literal["train", "test"], ...] = field(default=("train", "test"))
    save_name: str = field(default="data/adversarial_dataset/iterate-0/score-opt-350m/")
    batch_size: int = field(default=8)
    preprocessing_num_workers: Optional[int] = field(default=8)
    dataloader_num_workers: Optional[int] = field(default=8)
    template: Literal["default", "mistral"] = field(default="mistral")
    cutoff_len: int = field(default=2048)
    max_samples: Optional[int] = field(default=None)
    bf16: bool = field(default=False)
    flash_attn: Optional[str] = field(default="fa2")
    beta: Optional[float] = field(default=1)
    ref_model: str | None = field(default=None)

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
        
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature["{}_input_ids".format(key)],
                    "attention_mask": feature["{}_attention_mask".format(key)],
                    "labels": feature["{}_labels".format(key)],
                }
                if "pixel_values" in feature:
                    target_feature["pixel_values"] = feature["pixel_values"]

                if "{}_token_type_ids".format(key) in feature:
                    target_feature["token_type_ids"] = feature["{}_token_type_ids".format(key)]

                concatenated_features.append(target_feature)

        batch = super().__call__(concatenated_features)

        return batch, indices

def process_item(args):
    item, i, tokenizer, template, data_args = args
    results = []
    history = []
    for old_prompt, old_response in item["history"]:
        history.append({"role": Role.USER.value, "content": old_prompt})
        history.append({"role": Role.ASSISTANT.value, "content": old_response})
    for j, instruction in enumerate([item["instruction"]] + [x["prompt"] for x in item["generated_prompts"]]):
        prompt = history + [{"role": Role.USER.value, "content": instruction}]
        response = [
            {"role": Role.ASSISTANT.value, "content": item["chosen"]},
            {"role": Role.ASSISTANT.value, "content": item["rejected"]},
        ]
        system = ""
        tools = ""

        chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _encode_pairwise_example(
            prompt=prompt,
            response=response,
            system=system,
            tools=tools,
            template=template,
            tokenizer=tokenizer,
            processor=None,
            data_args=data_args,
        )
        
        result = {
            "index": (i, j - 1),
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": [1] * len(chosen_input_ids),
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": [1] * len(rejected_input_ids),
            "rejected_labels": rejected_labels,
        }
        results.append(result)
    return results

def gen(raw_data, tokenizer, template, data_args):
    args = [(item, i, tokenizer, template, data_args) for i, item in enumerate(raw_data)]
    with Pool(processes=data_args.preprocessing_num_workers) as pool:
        result_chunks = pool.map(process_item, args)

    for chunk in result_chunks:
        for result in chunk:
            yield result

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
            max_samples=args.max_samples,
            output_dir="dummy_dir",
            bf16=args.bf16,
            flash_attn=args.flash_attn,
            ref_model=args.ref_model,
        )
    )

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print("Arguments: ", args)

    device = accelerator.device

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=False)

    if args.ref_model is not None:
        ref_model = create_ref_model(model_args, finetuning_args)
        raise NotImplementedError()
    else:
        ref_model = None

    for split in args.dataset_splits:
        # Load raw data
        with open(os.path.join(args.raw_data_path, f"{split}.json"), "r") as f:
            raw_data = json.load(f)
        dataset = Dataset.from_generator(lambda: gen(raw_data, tokenizer, TEMPLATES.get(data_args.template, None), data_args))
        print(f"Found {len(dataset)} responses to calculate rewards in {args.raw_data_path}/{split}.json")

        # Prepare dataloader and model
        data_collator = PairwiseDataCollatorWithPadding(
            tokenizer=tokenizer, pad_to_multiple_of=8, label_pad_token_id=IGNORE_INDEX
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator, num_workers=args.dataloader_num_workers, pin_memory=True)
        model, dataloader = accelerator.prepare(model, dataloader)

        accelerator.wait_for_everyone()

        # Start calculating rewards
        change_list = []
        with torch.no_grad():
            for batch, indices in tqdm(dataloader, desc="Calculating rewards"):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits
                all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
                all_logps = all_logps / valid_length

                batch_size = batch["input_ids"].size(0) // 2
                chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)

                # chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
                # chosen_length, _ = valid_length.split(batch_size, dim=0)
            
                chosen_rewards = args.beta * chosen_logps.cpu().detach()
                rejected_rewards = args.beta * rejected_logps.cpu().detach()

                for index, chosen_reward, rejected_reward in zip(indices, chosen_rewards, rejected_rewards):
                    change_list.append([index, chosen_reward.item(), rejected_reward.item()])

        print(f"Rank {accelerator.local_process_index}: Processed {len(change_list)} items")
                    
        accelerator.wait_for_everyone()

        # Gather data across all devices, use gather_object if non-tensor data is involved
        gathered_change_list = accelerator.gather_for_metrics(change_list, use_gather_object=True)

        # Save the gathered data
        if accelerator.is_main_process:
            for index,  chosen_reward, rejected_reward in tqdm(gathered_change_list, desc="Processing gathered inputs"):
                i, j = index
                if j == -1:
                    raw_data[i]["chosen_reward"] = chosen_reward
                    raw_data[i]["rejected_reward"] = rejected_reward
                    raw_data[i]["chosen - rejected"] = chosen_reward - rejected_reward
                else:
                    raw_data[i]["generated_prompts"][j]["chosen_reward"] = chosen_reward
                    raw_data[i]["generated_prompts"][j]["rejected_reward"] = rejected_reward
                    raw_data[i]["generated_prompts"][j]["chosen - rejected"] = chosen_reward - rejected_reward

            for i, raw_item in enumerate(raw_data):
                raw_data[i]["generated_prompts"] = sorted(raw_item["generated_prompts"], key=lambda x: x["chosen - rejected"] if "chosen - rejected" in x else -100)
            
            save_path = os.path.join(args.save_name, f"{split}.json")
            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=4, ensure_ascii=False)
            print("Scored dataset have been saved at {}.".format(save_path))


if __name__ == "__main__":
    args = tyro.cli(ScriptArguments)
    main(args)