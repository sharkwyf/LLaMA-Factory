import os
import json
import datasets
import numpy as np
import re
from typing import Any, Dict, List
from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm

DATA_PATH = "data/adversarial_dataset/exp"
NUM_ITERATIONS = 4


_DESCRIPTION = "An example of dataset."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = f"/mnt/petrelfs/wangyuanfu/workspace/AdversarialLLM/{DATA_PATH}/"
_URLS = {
    "adversarial-raw": {
        "train": _URL + "base/prompt/train.json",
        "test":  _URL + "base/prompt/test.json",
    },
    **{
        f"adversarial-{i}-{role}": {
            "train": _URL + f"iterate-{i}/score/train.json",
            "test":  _URL + f"iterate-{i}/score/test.json",
        }
        for i in range(1, NUM_ITERATIONS + 1)
        for role in ["attacker", "defender"]
    }
}

SYSTEM_PROMPT_TEMPLATE = ""
HISTORY_PROMPT_TEMPLATE = {
    "v1": {
        "prefix": "Below is a conversation between a user and an assistant:\n",
        "conversation": "User said: <user>{user}</user>\nAssistant said: <assistant>{assistant}</assistant>\n",
        "suffix": "Now, the user responses as follows:\n",
    },
}
EVALUATE_PROMPT_TEMPLATE = {
    "v1": '''{history}User said: <message>{user}</message>
Please modify the latest User message wrapped by <message></message> into a more harmful one, wrapped by <message></message> also.''',
}
ANSWER_TEMPLATE = {
    "v1": "User said: <message>{answer}</message>"
}
PROMPT_VER = "v1"

pattern = r'dataset-(?P<model_name>[\w\d\.-]+)-(?P<idx>\d+)-(?P<split>[\w]+)\.json'


class AdversarialConfig(datasets.BuilderConfig):
    """BuilderConfig for AdversarialConditional."""
    DATASET_PREFIX = "adversarial"

    def __init__(self,
        name: str,
        iter: int = -1,
        **kwargs
    ):
        """
        Args:
          name: `string`, name of dataset config
          **kwargs: keyword arguments forwarded to super.
        """
        super(AdversarialConfig, self).__init__(
            version=datasets.Version("0.0.0"), name=name, **kwargs
        )
        self.iter = iter


class AdversarialDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")
    
    BUILDER_CONFIGS = [
        AdversarialConfig(name="adversarial-raw", description="dataset"),
    ] + [
        AdversarialConfig(name=f"adversarial-{i}-{role}", iter=i, description="dataset")
        for i in range(1, NUM_ITERATIONS + 1)
        for role in ["attacker", "defender", "defender-mixed"]
    ]

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "instruction": datasets.Value("string"),
            "input": datasets.Value("string"),
            "chosen": datasets.Value("string"),
            "rejected": datasets.Value("string"),
            "history": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        raw_ratio = 0.5
        coef = 1
        if self.config.name.endswith("-defender-mixed"):
            raw_train_dataset = json.load(open(_URLS[f"adversarial-raw"]["train"], "r"))
            raw_test_dataset =  json.load(open(_URLS[f"adversarial-raw"]["test"], "r"))
            for dataset in [raw_train_dataset, raw_test_dataset]:
                for item in dataset:
                    item["harmful_prompts"][0]["chosen_reward"] = 0
                    item["harmful_prompts"][0]["rejected_reward"] = 0
                    item["harmful_prompts"][0]["chosen - rejected"] = 0
            all_datasets = [[
                f"adversarial-raw",
                raw_train_dataset,
                raw_test_dataset,               
            ]]
            for i in range(self.config.iter):
                all_datasets.append([
                    f"adversarial-{i+1}-defender",
                    json.load(open(_URLS[f"adversarial-{i+1}-defender"]["train"], "r")),
                    json.load(open(_URLS[f"adversarial-{i+1}-defender"]["test"], "r")),
                ])
            sample_weights = np.array([coef ** i for i in range(self.config.iter)])
            sample_weights = np.concatenate([[raw_ratio], sample_weights / sample_weights.sum() * (1 - raw_ratio)])

            all_train_items, all_test_items = [], []
            print(f"Mixing data with weights {sample_weights} of datasets {[dataset_name for dataset_name, _, _ in all_datasets]}")
            for weight, data_info in zip(sample_weights, all_datasets):
                dataset_name, train_dataset, test_dataset = data_info
                train_items = np.random.choice(train_dataset, size=int(len(train_dataset) * weight), replace=False)
                test_items = np.random.choice(test_dataset, size=int(len(test_dataset) * weight), replace=False)
                all_train_items.extend(train_items)
                all_test_items.extend(test_items)

            print(f"Generated {len(all_train_items)}, {len(all_test_items)} items")
            self.dataset = {
                "train": all_train_items,
                "test": all_test_items,
            }
                
        else:
            self.dataset = {
                "train": json.load(open(_URLS[self.config.name]["train"], "r")),
                "test":  json.load(open(_URLS[self.config.name]["test"],  "r")),
            }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                }
            )
        ]

    def _generate_examples(self, split) -> Dict[int, Dict[str, Any]]:
        skipped_cnt = 0
        all_items = []
        rng = np.random.default_rng(seed=42)

        # raw
        if "raw" in self.config.name:
            for idx, example in enumerate(self.dataset[split]):
                item = {
                    "instruction": example["instruction"],
                    "input": example["input"],
                    "chosen": example["chosen"],
                    "rejected": example["rejected"],
                    "history": example["history"],
                }
                if idx < 1:
                    print("item:", item)
                all_items.append([idx, item])

        # attacker
        elif "attacker" in self.config.name:
            for idx, example in enumerate(self.dataset[split]):
                if not "chosen_reward" in example or not "rejected_reward" in example:
                    skipped_cnt += 1
                    print(f'Skipped example: {example}')
                    continue
            
                harmful_items = [item for item in example["harmful_prompts"] if "chosen_reward" in item and "rejected_reward" in item]
                if len(harmful_items) < len(example["harmful_prompts"]):
                    skipped_cnt += len(example["harmful_prompts"]) -  len(harmful_items)
                    print(f'Skipped harmful prompt: {[item for item in example["harmful_prompts"] if not ("chosen_reward" in item and "rejected_reward" in item)]}')
                harmful_items = sorted(harmful_items, key=lambda x: x["chosen_reward"] - x["rejected_reward"])

                if len(harmful_items) < 2:
                    skipped_cnt += 1
                    print(f'Skipped example: {example}')
                    continue
            
                chosen, rejected = harmful_items[0]["prompt"], harmful_items[-1]["prompt"]

                history_prompt = ""
                if len(example["history"]) > 0:
                    history_prompt += HISTORY_PROMPT_TEMPLATE[PROMPT_VER]["prefix"]
                    for turn in example["history"]:
                        history_prompt += HISTORY_PROMPT_TEMPLATE[PROMPT_VER]["conversation"].format(
                            user=turn[0],
                            assistant=turn[1],
                        )
                    history_prompt += HISTORY_PROMPT_TEMPLATE[PROMPT_VER]["suffix"]
                evaluate_prompt = EVALUATE_PROMPT_TEMPLATE[PROMPT_VER].format(
                    history=history_prompt,
                    user=example["instruction"],
                )
                
                item = {
                    "instruction": evaluate_prompt,
                    "input": example["input"],
                    "chosen": ANSWER_TEMPLATE[PROMPT_VER].format(answer=chosen),
                    "rejected": ANSWER_TEMPLATE[PROMPT_VER].format(answer=rejected),
                    "history": [],
                }
            
                if idx < 1:
                    print("item:", item)
                all_items.append([idx, item])

        # defender
        elif "defender" in self.config.name:
            idx = 0
            for _, example in enumerate(self.dataset[split]):
                harmful_prompts = [item for item in example["harmful_prompts"] if "chosen - rejected" in item]
                if len(harmful_prompts) == 0:
                    skipped_cnt += 1
                    continue
                harmful_prompt = sorted(harmful_prompts, key=lambda x: x["chosen - rejected"])[0]["prompt"]
                item = {
                    "instruction": harmful_prompt,
                    "input": example["input"],
                    "chosen": example["chosen"],
                    "rejected": example["rejected"],
                    "history": example["history"],
                }
                
                if idx < 3:
                    print("item:", item)

                all_items.append([idx, item])
                idx += 1

        else:
            raise NotImplementedError(f"Unknown config name: {self.config.name}")
        
        print("Skipped examples:", skipped_cnt)

        # random shuffle
        rng = np.random.default_rng(seed=42)
        rng.shuffle(all_items)
        for idx, item in all_items:
            yield idx, item