import json
import datasets
import numpy as np
from typing import Any, Dict, List
from datasets import load_dataset
from .template import ConditionalConfig


_DESCRIPTION = "An example of dataset."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/"
_URLS = {
    "train": {
        "helpful": [
            _URL + "helpful-base/train.jsonl.gz",
            _URL + "helpful-online/train.jsonl.gz",
            _URL + "helpful-rejection-sampled/train.jsonl.gz"
        ],
        "harmless": [
            _URL + "harmless-base/train.jsonl.gz",
        ],
    },
    "test": {
        "helpful": [
            _URL + "helpful-base/test.jsonl.gz",
            _URL + "helpful-online/test.jsonl.gz",
            _URL + "helpful-rejection-sampled/test.jsonl.gz"
        ],
        "harmless": [
            _URL + "harmless-base/test.jsonl.gz",
        ],
    },
}


class HHRLHFConfig(ConditionalConfig):
    """BuilderConfig for hh_rlhf."""
    DATASET_PREFIX = "hh_rlhf"
    PRINCIPLES = {"helpful", "harmless"}

    def __init__(self,
        name,
        principle="all",
        **kwargs
    ):
        """
        Args:
          name: `string`, name of dataset config
          **kwargs: keyword arguments forwarded to super.
        """
        super(HHRLHFConfig, self).__init__(
            version=datasets.Version("0.0.0"), name=name, dataset_prefix=self.DATASET_PREFIX, **kwargs
        )
        self.principle = principle
        if self.principle == "all":
            pass
        elif self.principle == "helpful":
            self.PRINCIPLES = {"helpful"}
        elif self.principle == "harmless":
            self.PRINCIPLES = {"harmless"}
        else:
            raise NotImplementedError()
        

class HHRLHFDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")
    
    BUILDER_CONFIGS = [
        HHRLHFConfig(name="all", principle="all", description="Overall dataset"),
        HHRLHFConfig(name="helpful", principle="helpful", description="Helpful dataset"),
        HHRLHFConfig(name="harmless", principle="harmless", description="Harmless dataset"),
        HHRLHFConfig(name="all_conditional", principle="all", description="Overall dataset",
            condition="prompt"),
        HHRLHFConfig(name="helpful_conditional", principle="helpful", description="Helpful dataset",
            condition="prompt"),
        HHRLHFConfig(name="harmless_conditional", principle="harmless", description="Harmless dataset",
            condition="prompt"),
        HHRLHFConfig(name="all_distribution", principle="all", description="Overall dataset",
            condition="distribution"),
        HHRLHFConfig(name="helpful_distribution", principle="helpful", description="Helpful dataset",
            condition="distribution"),
        HHRLHFConfig(name="harmless_distribution", principle="harmless", description="Harmless dataset",
            condition="distribution"),
        HHRLHFConfig(name="all_feature", principle="all", description="Overall dataset",
            condition="feature"),
        HHRLHFConfig(name="helpful_feature", principle="helpful", description="Helpful dataset",
            condition="feature"),
        HHRLHFConfig(name="harmless_feature", principle="harmless", description="Harmless dataset",
            condition="feature"),
    ]

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "instruction": datasets.Value("string"),
            "input": datasets.Value("string"),
            "chosen": datasets.Value("string"),
            "rejected": datasets.Value("string"),
            "history": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
            "principle": datasets.Value("string"),
            "key": datasets.Value("string"),
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        data = {}
        for split, item in _URLS.items():
            data[split] = {}
            for principle, urls in item.items():
                if principle in self.config.PRINCIPLES:
                    data[split][principle] = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data": data["train"],
                    "dataset_split": "train",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data": data["test"],
                    "dataset_split": "test",
                }
            )
        ]

    def _generate_examples(self, data, dataset_split):
        idx = 0
        all_items = []
        for principle, filepaths in data.items():
            for filepath in filepaths:
                with open(filepath, "r", encoding="utf-8") as f:
                    for row in f:
                        data = json.loads(row)
                        chosen = data["chosen"]
                        rejected = data["rejected"]

                        assist_idx = rejected.rfind("\n\nAssistant: ")
                        r_reject = rejected[assist_idx+13:].strip()
                        assist_idx = chosen.rfind("\n\nAssistant: ")
                        r_accept = chosen[assist_idx+13:].strip()

                        human_idx = chosen.rfind("\n\nHuman: ")
                        query = chosen[human_idx+9:assist_idx].strip()
                        prompt = chosen[:human_idx]
                        history = []

                        while prompt.rfind("\n\nAssistant: ") != -1:
                            assist_idx = prompt.rfind("\n\nAssistant: ")
                            human_idx = prompt.rfind("\n\nHuman: ")
                            if human_idx != -1:
                                old_query = prompt[human_idx+9:assist_idx].strip()
                                old_resp = prompt[assist_idx+13:].strip()
                                history.insert(0, (old_query, old_resp))
                            else:
                                break
                            prompt = prompt[:human_idx]

                        instruction, key = self.config.format_instruction(query, principle, principle, dataset_split=dataset_split)

                        item = {
                            "instruction": instruction,
                            "input": "",
                            "chosen": r_accept,
                            "rejected": r_reject,
                            "history": history,
                            "principle": principle,
                            "key": key,
                        }

                        if idx < 1:
                            print("item:", item)

                        all_items.append([idx, item])
                        idx += 1
        
        # random shuffle
        rng = np.random.default_rng(seed=42)
        rng.shuffle(all_items)
        for idx, item in all_items:
            yield idx, item
