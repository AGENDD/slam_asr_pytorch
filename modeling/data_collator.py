from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from torch import Tensor
import numpy as np

@dataclass
class DataCollatorForSlamASR(object):
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Tensor]:
        # Extract elements
        # x = ds["speech"][0:3]
        # y = ds["text"][0:3]

        x = [np.array(i["speech"]) for i in instances]
        y = [i["translation"].lower() for i in instances]
        z = [i["prompt"].lower() for i in instances]
        return {"audios": x, "prompts": z, "transcriptions": y}
