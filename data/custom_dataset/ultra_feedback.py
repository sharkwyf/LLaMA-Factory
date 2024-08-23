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
_URL = "openbmb/UltraFeedback"


class UltraFeedbackConfig(ConditionalConfig):
    """BuilderConfig for UltraFeedback."""
    DATASET_PREFIX = "ultra_feedback"
    PRINCIPLES = ["overall", "helpfulness", "honesty", "instruction_following", "truthfulness"]

    def __init__(self,
        name: str,
        sampling_method="random",
        principles_sample_prob=[1, 0, 0, 0, 0],
        **kwargs
    ):
        """
        Args:
          name: `string`, name of dataset config
          **kwargs: keyword arguments forwarded to super.
        """
        super(UltraFeedbackConfig, self).__init__(
            version=datasets.Version("0.0.0"), name=name, dataset_prefix=self.DATASET_PREFIX, **kwargs
        )
        self.sampling_method = sampling_method
        self.principles_sample_prob = np.array(principles_sample_prob) / np.sum(principles_sample_prob)


class UltraFeedbackDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")
    
    BUILDER_CONFIGS = [
        UltraFeedbackConfig(name="all", description="all dataset",
            principles_sample_prob=[0.2, 0.2, 0.2, 0.2, 0.2]),
        UltraFeedbackConfig(name="overall", description="overall dataset",
            principles_sample_prob=[1, 0, 0, 0, 0]),
        UltraFeedbackConfig(name="helpfulness", description="helpfulness dataset",
            principles_sample_prob=[0, 1, 0, 0, 0]),
        UltraFeedbackConfig(name="honesty", description="honesty dataset",
            principles_sample_prob=[0, 0, 1, 0, 0]),
        UltraFeedbackConfig(name="instruction_following", description="instruction_following dataset",
            principles_sample_prob=[0, 0, 0, 1, 0]),
        UltraFeedbackConfig(name="truthfulness", description="truthfulness dataset",
            principles_sample_prob=[0, 0, 0, 0, 1]),
        UltraFeedbackConfig(name="all_conditional", description="all dataset",
            principles_sample_prob=[0.2, 0.2, 0.2, 0.2, 0.2], condition="prompt"),
        UltraFeedbackConfig(name="overall_conditional", description="overall dataset",
            principles_sample_prob=[1, 0, 0, 0, 0], condition="prompt"),
        UltraFeedbackConfig(name="helpfulness_conditional", description="helpfulness dataset",
            principles_sample_prob=[0, 1, 0, 0, 0], condition="prompt"),
        UltraFeedbackConfig(name="honesty_conditional", description="honesty dataset",
            principles_sample_prob=[0, 0, 1, 0, 0], condition="prompt"),
        UltraFeedbackConfig(name="instruction_following_conditional", description="instruction_following dataset",
            principles_sample_prob=[0, 0, 0, 1, 0], condition="prompt"),
        UltraFeedbackConfig(name="truthfulness_conditional", description="truthfulness dataset",
            principles_sample_prob=[0, 0, 0, 0, 1], condition="prompt"),
        UltraFeedbackConfig(name="all_distribution", description="all dataset",
            principles_sample_prob=[0.2, 0.2, 0.2, 0.2, 0.2], condition="distribution"),
        UltraFeedbackConfig(name="overall_distribution", description="overall dataset",
            principles_sample_prob=[1, 0, 0, 0, 0], condition="distribution"),
        UltraFeedbackConfig(name="helpfulness_distribution", description="helpfulness dataset",
            principles_sample_prob=[0, 1, 0, 0, 0], condition="distribution"),
        UltraFeedbackConfig(name="honesty_distribution", description="honesty dataset",
            principles_sample_prob=[0, 0, 1, 0, 0], condition="distribution"),
        UltraFeedbackConfig(name="instruction_following_distribution", description="instruction_following dataset",
            principles_sample_prob=[0, 0, 0, 1, 0], condition="distribution"),
        UltraFeedbackConfig(name="truthfulness_distribution", description="truthfulness dataset",
            principles_sample_prob=[0, 0, 0, 0, 1], condition="distribution"),
        UltraFeedbackConfig(name="all_feature", description="all dataset",
            principles_sample_prob=[0.2, 0.2, 0.2, 0.2, 0.2], condition="feature"),
        UltraFeedbackConfig(name="overall_feature", description="overall dataset",
            principles_sample_prob=[1, 0, 0, 0, 0], condition="feature"),
        UltraFeedbackConfig(name="helpfulness_feature", description="helpfulness dataset",
            principles_sample_prob=[0, 1, 0, 0, 0], condition="feature"),
        UltraFeedbackConfig(name="honesty_feature", description="honesty dataset",
            principles_sample_prob=[0, 0, 1, 0, 0], condition="feature"),
        UltraFeedbackConfig(name="instruction_following_feature", description="instruction_following dataset",
            principles_sample_prob=[0, 0, 0, 1, 0], condition="feature"),
        UltraFeedbackConfig(name="truthfulness_feature", description="truthfulness dataset",
            principles_sample_prob=[0, 0, 0, 0, 1], condition="feature"),
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
        self.dataset = load_dataset(
            path=_URL,
            split="train",
        )
        self.dataset = self.dataset.train_test_split(test_size=0.1, seed=42)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "dataset_split": "train",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "dataset_split": "test",
                }
            )
        ]

    def _generate_examples(self, split, dataset_split) -> Dict[int, Dict[str, Any]]:
        all_items = []
        for idx, example in enumerate(self.dataset[split]):
            # Sample a principle
            principle = np.random.choice(self.config.PRINCIPLES, p=self.config.principles_sample_prob)
            
            # Sort the chosen completions by score in descending order
            sorted_completions = sorted(
                example["completions"],
                key=lambda x: x['annotations'][principle]["Rating"] if principle in x['annotations'] else x["overall_score"],
                reverse=True)

            if len(sorted_completions) < 2:
                print("skipping example", idx, "because it has less than 2 completions")
                continue

            if self.config.sampling_method == 'random':
                # randomly choose two completions from all completions in data item without change the sorted order
                chosen_indices = sorted(np.random.choice(len(sorted_completions), 2, replace=False))
                chosen_completions = [sorted_completions[i] for i in chosen_indices]
                output = [comp['response'] for comp in chosen_completions]
            elif self.config.sampling_method == "highlow":
                # choose the highest and lowest scoring completions
                chosen_completions = [sorted_completions[0], sorted_completions[-1]]
                output = [comp['response'] for comp in chosen_completions]
            else:
                raise NotImplementedError()
            
            instruction, key = self.config.format_instruction(example["instruction"], principle, principle, dataset_split=dataset_split)
            chosen, rejected = output

            item = {
                "instruction": instruction,
                "input": "",
                "chosen": chosen,
                "rejected": rejected,
                "history": [],
                "principle": principle,
                "key": key,
            }

            if idx < 1:
                print("item:", item)
                
            all_items.append([idx, item])
        
        # random shuffle
        rng = np.random.default_rng(seed=42)
        rng.shuffle(all_items)
        for idx, item in all_items:
            yield idx, item