from datasets import load_dataset
# import soundfile as sf
import torch
from modeling.asr import SLAM_ASR
from safetensors.torch import load_file
from datasets import load_from_disk
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
ds = ds['test'].select(range(100))
ds = ds.map(map_to_array)

for i in range(len(ds)):
    x = ds[i]["speech"]
    y = ds[i]["translation"]
    pr = ds[i]["prompt"]
    # asr(x)

    output = asr.generate(x, pr)  # causal of shape (b, seq_len, vocab_size)
    print(f"Predicted: {asr.language_tokenizer.batch_decode(output)[0]}")
    print(f"Reference: {y}")
    print("\n\n")
