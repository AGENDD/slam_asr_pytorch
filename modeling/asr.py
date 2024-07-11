"""
The main body of the ASR model,

User: <Speech> <Prompt>
Model: <Transcription>
"""

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List

try:
    from .speech_encoder import SpeechEncoder
except ImportError:
    from speech_encoder import SpeechEncoder

from transformers import AutoModelForCausalLM, AutoTokenizer

from .lora import LinearWithLoRA


class SLAM_ASR(nn.Module):
    def __init__(
        self,
        speech_encoder_model_id,
        language_model_id,
        downsample_K=5,
        hidden_dim=2048,
        train_mode="adapter",
        device="cuda",
        token = "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",
    ):
        assert train_mode in ["adapter", "full"]
        super(SLAM_ASR, self).__init__()
        self.device = device
        """
                       |------|
                       |hubert|
                       |------|
                           |
                           |
                       (adapter, the only trainable part here)
                           |
                           v
        LLAMA("<user>: [______], transcribe it, <assistant>") --> "The weather is good"

        """

        self.language_tokenizer = AutoTokenizer.from_pretrained(language_model_id,trust_remote_code=True,pad_token="<s>")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_id,
            trust_remote_code=True,
            # token = token
        ).to(self.device)
        # self.language_model = AutoModelForCausalLM.from_pretrained("temp_models/rwkv-6-world-1b6", trust_remote_code=True)
        
        # self.language_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.language_model.resize_token_embeddings(len(self.language_tokenizer))
        
        ###
        # print("Model before lora:")
        # print(self.language_model)
        # self.load_lora(self.language_model)
        # print("Model after lora:")
        # print(self.language_model)
        ###
                
        language_project_dim = self.language_model.config.hidden_size

        self.speech_encoder = SpeechEncoder(
            speech_encoder_model_id,
            language_project_dim,
            downsample_K=downsample_K,
            hidden_dim=hidden_dim,
            train_mode=train_mode,
            device=device,
        ).to(self.device)
      
        
        # print("show language params")
        # for name,param in self.language_model.named_parameters():
        #     print(f"{name}:{param.requires_grad}")
        # print("show language model")
        # print(self.language_model)


        self.set_gradient(train_mode)
        

        

        # self.prompt_part1 = """User:"""
        # self.prompt_part2 = (
        #     """transcribe it.\nAssistant:"""
        # )
        # self.embed_bank = {"embed1": None, "embed2": None, "att1": None, "att2": None}
        # self.set_embed_bank()

    def gradient_checkpointing_enable(self, **kwargs):
        self.language_model.gradient_checkpointing_enable(**kwargs)

    # def load_lora(self, model):
    #     for name, child in model.named_children():
    #         if isinstance(child, nn.Linear):
    #             new_layer = LinearWithLoRA(child, 32,self.device)
    #             delattr(model, name)
    #             model.add_module(name, new_layer)
    #         else:
    #             self.load_lora(child)
                
    def load_lora(self, model):
        to_replace = []
        for name, child in model.named_children():
            if isinstance(child, nn.Linear):
                to_replace.append((name, child))
            else:
                self.load_lora(child)
        for name, child in to_replace:
            new_layer = LinearWithLoRA(child, 128, self.device)
            # new_layer.print_parameters()
            delattr(model, name)
            model.add_module(name, new_layer)
        for param in model.parameters():
            pass
    
    def set_embed_bank(self, batch_size=1):
        input_dict1 = self.language_tokenizer(
            [self.prompt_part1], return_tensors="pt"
        ).to(self.device)
        input_dict2 = self.language_tokenizer(
            [self.prompt_part2], return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        # precache the embeddings for prompt
        with torch.no_grad():
            inputs_embeds1 = self.language_model.rwkv.get_input_embeddings()(
                input_dict1.input_ids
            )
            inputs_embeds2 = self.language_model.rwkv.get_input_embeddings()(
                input_dict2.input_ids
            )
        self.embed_bank["embed1"] = inputs_embeds1
        self.embed_bank["embed2"] = inputs_embeds2
        self.embed_bank["att1"] = input_dict1.attention_mask
        self.embed_bank["att2"] = input_dict2.attention_mask
        print("[Preloaded embeddings for both part of the prompt.]")
        print(
            f"    {self.prompt_part1}        {inputs_embeds1.shape}\n    {self.prompt_part2}        {inputs_embeds2.shape}"
        )

    def set_gradient(self, train_mode):
        assert train_mode in ["adapter", "full"]

        # call set_gradient for speech encoder
        self.speech_encoder.set_gradient(train_mode)

        # freeze the whole language_model
        if train_mode != "full":
            for name, param in self.language_model.named_parameters():
                # print(f"layer:{name}")
                
                if('lora' not in name.lower()):
                    param.requires_grad = False
            
        # now list all parameters that require grad
        print("Parameters that require grad:")

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"    {name}: {param.shape}")

    
    def remove_padding(self, x, mask):
                #去除speech_output的padding部分
        x_no_padding = []
        # 对于每一个样本和对应的掩码
        for x_i, mask_i in zip(x, mask):
            # 使用掩码来选择非填充部分
            x_i_no_padding = x_i[mask_i.bool()]
            # 将结果添加到列表中
            x_no_padding.append(x_i_no_padding)
        
        return x_no_padding
    
    def concatenate_audio_transcription(self, audio, transcription):
        #将没有padding的audio和transcirption在第二维度拼起来
        result = []
        for sublist1, sublist2 in zip(audio, transcription):
            sub_result = torch.cat((sublist1 ,sublist2), dim=0)
            result.append(sub_result)

        return result
    
    
    def _prepare_input_embeds(
        self, audios: List[float], transcriptions: List[str] = None
    ):
        """
        First, run audios through speech_encoder to get the embeddings and mask
        """
        
        
        speech_output, mask = self.speech_encoder(audios) #no padding
        
        print(f"audio after hubert and adapter:\t{speech_output.shape}")
        print(f"audio mask:\t{mask.shape}")
        
        
        audio_no_padding = self.remove_padding(speech_output,mask)
        print(f"audio with no padding:\t{len(audio_no_padding)}-{[len(x) for x in audio_no_padding]}")
        
        
        # batch_size = speech_output.shape[0]
        # get the prompt embeddings
        # embed1 = self.embed_bank["embed1"].repeat(batch_size, 1, 1)  # (b, 4, 2048)
        # embed2 = self.embed_bank["embed2"].repeat(batch_size, 1, 1)  # (b, 11, 2048)
        # att1 = self.embed_bank["att1"].repeat(batch_size, 1)
        # att2 = self.embed_bank["att2"].repeat(batch_size, 1)

        true_labels = None
        
        ##################################
        # pr_output = self.language_tokenizer(
        #     prompts, return_tensors="pt", add_special_tokens=False,padding=True
        # ).to(self.device)
        # att_pr = pr_output.attention_mask
        # with torch.no_grad():
        #     pr_output = self.language_model.model.embed_tokens(
        #         pr_output.input_ids
        #     )
        ##################################
        
        # prompt_length = embed1.shape[1] + speech_output.shape[1] +  embed2.shape[1]
        
        
        
        if transcriptions is not None:
            _labels = self.language_tokenizer(
                transcriptions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ).to(self.device)
            
            labels_embeds = self.language_model.rwkv.get_input_embeddings()(_labels.input_ids)
            att3 = _labels.attention_mask
            padding_dim = labels_embeds[0][len(labels_embeds[0]) - 1]
            
            print(f"padding_dim:\t{padding_dim.shape}")
            print(f"embed transcription:\t{labels_embeds.shape}")
            print(f"transcription mask:\t{att3.shape}")
            
            label_no_padding = self.remove_padding(labels_embeds, att3)
            
            print(f"embed transcription with no padding:\t{len(label_no_padding)}-{[len(x) for x in label_no_padding]}")
            
            audio_label = self.concatenate_audio_transcription(audio_no_padding , label_no_padding)
            print(f"concatenated inputs with no padding:\t{len(audio_label)}-{[len(x) for x in audio_label]}")
            
            exit(0)

            prompt_embed = torch.cat(
                [speech_output, labels_embeds], dim=1
            )  # (b, 4+audio+11+seq_len, 2048)
            prompt_mask = torch.cat([mask, att3], dim=1)

            true_labels = _labels.input_ids
            # attach "prompt_length" * -100 to the start of the true_labels
            true_labels = torch.cat(
                [
                    torch.full(
                        (batch_size),
                        -100,
                        dtype=torch.long,
                        device=self.device,
                    ),
                    true_labels,
                ],
                dim=1,
            )
        else:
            exit(0)
            # prompt_embed = torch.cat(
            #     [embed1, speech_output, embed2], dim=1
            # )  # (b, 4+audio+11, 2048)
            # prompt_mask = torch.cat([att1, mask, att2], dim=1)
            true_labels = None
        return prompt_embed, prompt_mask, true_labels

    def forward(self, audios: List[float], transcriptions: List[str] = None):
        
        
        
        prompt_embed, prompt_mask, true_labels = self._prepare_input_embeds(
            audios, transcriptions
        )
        # run the prompt through the language model

        outputs = self.language_model(
            inputs_embeds=prompt_embed,
            attention_mask=prompt_mask.bool(),
            labels=true_labels,
        )  # CausalLMOutputWithPast
        
        # print(f"true_labels:{true_labels[0]}-outputs:{true_labels[0]}",end="\r")
        return outputs

    def generate(self, audios: List[float], stopping_criteria=None):
        """
        Generate the transcription
        """
        prompt_embed, prompt_mask, _ = self._prepare_input_embeds(audios)
        
        outputs = self.language_model(
            inputs_embeds=prompt_embed,
            attention_mask=prompt_mask.bool()
        )
        return outputs

    @property
    def config(self):
        return self.language_model.config
