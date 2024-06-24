from transformers import LlamaForCausalLM, LlamaTokenizer



language_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"

tokenizer = LlamaTokenizer.from_pretrained(language_model_id)


model = LlamaForCausalLM.from_pretrained(
            language_model_id,
            trust_remote_code=True,
        )

print(model)