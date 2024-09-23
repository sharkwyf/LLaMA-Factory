import os
import json
import datasets
import numpy as np
import re
from typing import Any, Dict, List
from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm


_DESCRIPTION = "An example of dataset."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = f"./data/relabeled_dataset/"
_URLS = {
    "hh_rlhf_harmless": {
        "train": _URL + "hh_rlhf_harmless/train.json",
        "test":  _URL + "hh_rlhf_harmless/test.json",
    },
}


class RelabledDatasetConfig(datasets.BuilderConfig):
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
        super(RelabledDatasetConfig, self).__init__(
            version=datasets.Version("0.0.0"), name=name, **kwargs
        )
        self.iter = iter


class RelabledDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")
    
    BUILDER_CONFIGS = [
        RelabledDatasetConfig(name="hh_rlhf_harmless", description="dataset"),
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
        all_items = []
        relabeled_cnt = 0
        skipped_cnt = 0

        for idx, example in enumerate(self.dataset[split]):
            item = {
                "instruction": example["instruction"],
                "input": example["input"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
                "history": example["history"],
            }
            if not "chosen_score" in example or not "rejected_score" in example:
                skipped_cnt += 1
                continue
            if example["chosen_score"] < example["rejected_score"]:
                item["chosen"], item["rejected"] = example["rejected"], example["chosen"]
                relabeled_cnt += 1

            if idx < 1:
                print("item:", item)

            all_items.append([idx, item])
        
        print(f"Relabeled {relabeled_cnt} examples out of {len(all_items)} in total, skipped {skipped_cnt} items.")
       
        # random shuffle
        rng = np.random.default_rng(seed=42)
        rng.shuffle(all_items)
        for idx, item in all_items:
            yield idx, item