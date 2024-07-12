import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling.asr import SLAM_ASR
from datasets import load_from_disk
# model = AutoModelForCausalLM.from_pretrained("temp_models/rwkv-6-world-1b6", trust_remote_code=True, torch_dtype=torch.float16).to(0)
# tokenizer = AutoTokenizer.from_pretrained("temp_models/rwkv-6-world-1b6", trust_remote_code=True, padding_side='left', pad_token="<s>")
token = "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"

model = SLAM_ASR(
    speech_encoder_model_id ="facebook/hubert-base-ls960",
    # language_model_id="openlm-research/open_llama_3b",
    language_model_id="temp_models/rwkv-6-world-1b6",
    train_mode="adapter",
    token = token,
)

# data = load_from_disk("temp_datasets/en-final").select(range(3))

# output = model([i["speech"] for i in data], [i["text"].lower() for i in data])
for _, module in model.language_model.named_modules():
    for name, param in module.named_parameters():
        print(f"layer:{name}")
        
        if('state' not in name.lower()):
            param.requires_grad = False
        else:
            print("yesy")
            param.requires_grad = True
