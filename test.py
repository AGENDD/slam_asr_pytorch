<<<<<<< Updated upstream
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("temp_models/rwkv-6-world-1b6", trust_remote_code=True, torch_dtype=torch.float16).to(0)
tokenizer = AutoTokenizer.from_pretrained("temp_models/rwkv-6-world-1b6", trust_remote_code=True, padding_side='left', pad_token="<s>")

print(tokenizer.eos_token)
print(tokenizer.bos_token)
=======
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import soundfile as sf

import pandas as pd
import importlib
from packaging import version
from packaging.version import parse

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    set_seed,
    Seq2SeqTrainer,
)
from datasets import load_dataset, Dataset
import evaluate
from modeling.arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    GenerationArguments,
)
from modeling.data_collator import DataCollatorForSlamASR
from modeling.asr import SLAM_ASR
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True




def format_dataset(dataset):
    def map_to_array(batch):
        speech, _ = sf.read(batch["file"])
        batch["speech"] = speech
        return batch

    dataset = dataset.map(map_to_array)
    print(dataset)
    # print(dataset.column_names)
    dataset = dataset.remove_columns(
        [
            col
            for col in dataset.column_names["train"]
            if col not in ["speech", "text"]
        ]
    )
    return dataset

dataset = load_dataset("librispeech_asr", "clean", trust_remote_code=True)
# rename TRAIN_TAG to "train"
dataset["train"] = dataset.pop("train.360")
dataset = format_dataset(dataset)
# print(
#     f"Splitting train dataset in train and validation according to `eval_dataset_size = {args.eval_dataset_size}`"
# )
dataset = dataset["train"].train_test_split(
    test_size=args.eval_dataset_size, shuffle=True, seed=42
)
# Split train/eval, reduce size
# if args.do_eval or args.do_predict:
#     if "validation" in dataset:
#         eval_dataset = dataset["validation"]
#     else:
#         eval_dataset = dataset["test"]
#     if (
#         args.max_eval_samples is not None
#         and len(eval_dataset) > args.max_eval_samples
#     ):
#         eval_dataset = eval_dataset.select(range(args.max_eval_samples))
#     if args.group_by_length:
#         eval_dataset = eval_dataset.map(lambda x: {"length": len(x["text"])})
# if args.do_train:
if True:
    train_dataset = dataset["train"]
    # if (
    #     args.max_train_samples is not None
    #     and len(train_dataset) > args.max_train_samples
    # ):
    train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.group_by_length:
        train_dataset = train_dataset.map(lambda x: {"length": len(x["text"])})
# data_collator = DataCollatorForSlamASR(
#     source_max_len=args.source_max_len,
#     target_max_len=args.target_max_len,
#     train_on_source=args.train_on_source,
#     predict_with_generate=args.predict_with_generate,
# )

print(dict(
    train_dataset=train_dataset,
    # None,
    # None,
    # eval_dataset=eval_dataset if args.do_eval else None,
    # predict_dataset=eval_dataset if args.do_predict else None,
    # data_collator=data_collator,
))

# return dict(
#     train_dataset=train_dataset if args.do_train else None,
#     # eval_dataset=eval_dataset if args.do_eval else None,
#     # predict_dataset=eval_dataset if args.do_predict else None,
#     data_collator=data_collator,
# )
    
 
# data_module = make_data_module(tokenizer=tokenizer, args=args)
>>>>>>> Stashed changes
