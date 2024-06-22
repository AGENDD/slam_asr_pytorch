# from datasets import load_from_disk
# from datasets import DatasetDict

# dataset = load_from_disk("covost_en2zh-CN")

# new_data = DatasetDict({
#     'train':dataset['train'].select(range(2000)),
#     'validation':dataset['validation'].select(range(200)),
#     'test':dataset['test'].select(range(200))
# })

# new_data.save_to_disk("covost_en2zh-CN-tiny")

from datasets import load_from_disk
from datasets import DatasetDict
import numpy as np
dataset = load_from_disk("covost_zh-CN2en-final")

print(dataset)
for i in range(10):
    print(dataset[i]['translation'])
    print(dataset[i]['prompt'])
    print(dataset[i]['speech'][:20])

from datasets import load_dataset
dataset = load_dataset("librispeech_asr")

dataset = DatasetDict(
    {
        "train": dataset["train.360"],
        "validation": dataset["validation"],
        "test": dataset["test"],
    }
)



