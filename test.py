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

dataset = load_from_disk("covost_en2zh-CN-tiny")
print(dataset["train"][0]["audio"]['array'].shape)
print(dataset["train"][1]["audio"]['array'].shape)
print(dataset["train"][2]["audio"]['array'].shape)
print(dataset["train"][3]["audio"]['array'].shape)
print(dataset["train"][4]["audio"]['array'].shape)