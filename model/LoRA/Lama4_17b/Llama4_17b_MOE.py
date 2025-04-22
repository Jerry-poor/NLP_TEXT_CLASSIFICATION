import os
import json
import torch
import torch.nn as nn
from transformers import AutoConfig, BitsAndBytesConfig
from transformers import LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, get_peft_model


class MoEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # routing network
        self.router = nn.Linear(config.hidden_size, config.num_experts_per_tok, bias=False)
        # experts: down -> activation -> up
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            )
            for _ in range(config.num_experts_per_tok)
        ])

    def forward(self, x):  # x: [batch, seq, hidden]
        scores = torch.softmax(self.router(x), dim=-1)              # [B,S,E]
        expert_outs = torch.stack([e(x) for e in self.experts], dim=-1)  # [B,S,H,E]
        return torch.einsum("bsh e, bshf e -> bshf", scores, expert_outs)  # [B,S,H]


class Llama4MoEForCausalLM(LlamaPreTrainedModel):
    config_class = None
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        # replace each layer's MLP with MoE module
        for layer in self.model.layers:
            layer.mlp = MoEFeedForward(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def get_moe_model(
    model_name: str,
    quant_config: BitsAndBytesConfig,
    lora_config: LoraConfig,
) -> Llama4MoEForCausalLM:
    """
    Load and return a quantized, MoE-enabled Llama-4 model wrapped with LoRA.
    """
    # load raw JSON config
    raw_cfg_path = os.path.join(model_name, "config.json")
    raw_cfg = json.load(open(raw_cfg_path, "r"))
    text_cfg = raw_cfg.get("text_config", {})

    # base hf config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    # fill missing fields
    config.vocab_size = text_cfg["vocab_size"]
    config.hidden_size = text_cfg["hidden_size"]
    config.num_hidden_layers = text_cfg["num_hidden_layers"]
    config.num_attention_heads = text_cfg["num_attention_heads"]
    config.intermediate_size = text_cfg["intermediate_size"]
    config.rms_norm_eps = text_cfg["rms_norm_eps"]
    config.num_key_value_heads = text_cfg["num_key_value_heads"]
    config.attention_dropout = text_cfg["attention_dropout"]
    config.attention_bias = text_cfg["attention_bias"]
    config.mlp_bias = False
    config.rope_theta = text_cfg["rope_theta"]
    config.initializer_range = text_cfg["initializer_range"]
    config.num_experts_per_tok = text_cfg["num_experts_per_tok"]

    # instantiate and quantize
    model = Llama4MoEForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=False,
    )
    # apply LoRA
    model = get_peft_model(model, lora_config)
    return model
