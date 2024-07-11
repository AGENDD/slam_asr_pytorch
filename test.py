import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("temp_models/rwkv-6-world-1b6", trust_remote_code=True, torch_dtype=torch.float16).to(0)
tokenizer = AutoTokenizer.from_pretrained("temp_models/rwkv-6-world-1b6", trust_remote_code=True, padding_side='left', pad_token="<s>")

print(tokenizer.eos_token)
print(tokenizer.bos_token)
