import numpy as np
from datasets import Dataset, concatenate_datasets, interleave_datasets, load_dataset, load_from_disk

NUM_PROC = 1

# data_path = "/mnt/petrelfs/wangyuanfu/workspace/AdversarialLLM/data/adversarial_dataset/adversarial_dataset.py"
# subset = "adversarial-raw"
# dataset1 = load_dataset(
#     path=data_path,
#     name=subset,
#     trust_remote_code=True,
#     num_proc=NUM_PROC
# )
# print("dataset", dataset1)

# data_path = "/mnt/petrelfs/wangyuanfu/workspace/AdversarialLLM/data/adversarial_dataset/adversarial_dataset.py"
# subset = "adversarial-1-attacker"
# dataset2 = load_dataset(
#     path=data_path,
#     name=subset,
#     trust_remote_code=True,
#     num_proc=NUM_PROC
# )
# print("dataset", dataset2)

data_path = "/mnt/petrelfs/wangyuanfu/workspace/AdversarialLLM/data/adversarial_dataset/adversarial_dataset.py"
subset = "adversarial-3-defender-mixed"
dataset3 = load_dataset(
    path=data_path,
    name=subset,
    trust_remote_code=True,
    num_proc=NUM_PROC
)
print("dataset", dataset3)

print("Done")
