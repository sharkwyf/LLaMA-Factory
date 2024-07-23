import numpy as np
import datasets
import torch

_FORMAT_INSTRUCTION_PROMPT="{principle} - {instruction}"


class ConditionalConfig(datasets.BuilderConfig):
    def __init__(self,
        condition: str = None,
        dataset_prefix: str = None,
        **kwargs
    ):
        """
        Args:
          name: `string`, name of dataset config
          **kwargs: keyword arguments forwarded to super.
        """
        super(ConditionalConfig, self).__init__(**kwargs)
        self.condition = condition
        self.dataset_prefix = dataset_prefix
    
    def format_instruction(self, instruction, principle, idx, dataset_split):
        if self.condition == "prompt":
            instruction = _FORMAT_INSTRUCTION_PROMPT.format(
                instruction=instruction,
                principle=principle,
            )
        else:
            instruction = instruction
        key = f"{self.dataset_prefix}_{idx}_{dataset_split}_feature"
        return instruction, key
