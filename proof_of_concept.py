from datasets import load_dataset
# import soundfile as sf
import torch
from modeling.asr import SLAM_ASR
from safetensors.torch import load_file
from datasets import load_from_disk

import soundfile as sf
import numpy as np
import os
from playsound import playsound


torch.cuda.set_device(1)

asr = SLAM_ASR(
    speech_encoder_model_id ="facebook/hubert-base-ls960",
        # language_model_id="TinyLlama/TinyLlama-1.1B-Chat-v0.4",
    language_model_id="openlm-research/open_llama_3b",
    train_mode="adapter",
)
# load the state_dict from output/adapter_weights.pt
adapter_weight = load_file("output/covost_slam_asr_ch2en/checkpoint-16400/model.safetensors")
asr.load_state_dict(adapter_weight, strict=False)


def map_to_array(batch):
    
    batch["speech"] = batch["audio"]['array'][0]
    return batch


ds = load_from_disk(
    "temp_datasets/covost_zh-CN2en"
)
ds = ds['validation'].select(range(100))
ds = ds.map(map_to_array)

# with open("temp_audio/text.txt",'w') as f:
for i in range(len(ds)):
    x = ds[i]["speech"]
    y = ds[i]["translation"]
    pr = ds[i]["prompt"]
    z = ds[i]["sentence"]
    # asr(x)
    
    

    output = asr.generate(x, pr)  # causal of shape (b, seq_len, vocab_size)
    output = asr.language_tokenizer.batch_decode(output)[0]
    output = output.replace("[PAD]","")
    print(f"Predicted: {output}")
    print(f"Reference: {y}")
    print(f"Source:{z}")
    print("\n\n")
        
        # f.write(f"Predicted: {output}\n")
        # f.write(f"Reference: {y}\n")
        # f.write(f"Source:{z}")
        # f.write("\n\n")
        
        # sf.write(f'temp_audio/temp{i}.wav', x, 16000)
        # playsound('temp.wav')
        # os.remove('temp.wav')
