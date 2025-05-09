# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import torch
from einops import rearrange
from safetensors.torch import save_file

transformer_configs = {
    "2B": dict(n_layer=30, n_head=20, dim=2560, vocab_size=128256, n_local_heads=5, intermediate_size=6912),
}


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    block_size: int = 4096
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config in str(name).upper() or config in str(name)]
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])


@torch.inference_mode()
def convert_ts_checkpoint(
    *,
    input_file: str = "",
    output_dir: str = "",
    model_name: Optional[str] = None,
) -> None:

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    weight_map = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "layers.{}.attention.wqkv.weight": "model.layers.{}.self_attn.q_proj.weight",
        "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
        'layers.{}.feed_forward.w13.weight': 'model.layers.{}.mlp.gate_proj.weight',
        "layers.{}.feed_forward.w2.weight": "model.layers.{}.mlp.down_proj.weight",
        "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
        "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
        'layers.{}.feed_forward.ffn_sub_norm.weight': 'model.layers.{}.mlp.ffn_sub_norm.weight',
        "layers.{}.attention.attn_sub_norm.weight": "model.layers.{}.self_attn.attn_sub_norm.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }

    def convert_q(weight, config):
        weight = rearrange(weight, '(h d l) i -> (h l d) i', l=2, h=config.n_head)
        return weight
    
    def convert_k(weight, config):
        weight = rearrange(weight, '(h d l) i -> (h l d) i', l=2, h=config.n_local_heads)
        return weight

    def quant_weight_fp16(weight):
        s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
        new_weight = (weight * s).round().clamp(-1, 1) / s
        return new_weight

    merged_result = {}
    for file in [input_file]:
        state_dict = torch.load(str(file), map_location="cpu", mmap=True)
        merged_result.update(state_dict)
    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key, 1)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]
        
        if 'wqkv' in key:
            wq = value[:config.dim]
            wk = value[config.dim:config.head_dim * config.n_local_heads + config.dim]
            wv = value[config.head_dim * config.n_local_heads + config.dim:]
            wq = convert_q(wq, config)
            wk = convert_k(wk, config)
            final_result[new_key] = quant_weight_fp16(wq)
            final_result[new_key.replace('q_proj', 'k_proj')] = quant_weight_fp16(wk)
            final_result[new_key.replace('q_proj', 'v_proj')] = quant_weight_fp16(wv)
        elif 'w13' in key:
            w1 = value[:config.intermediate_size]
            w3 = value[config.intermediate_size:]
            final_result[new_key] = quant_weight_fp16(w1)
            final_result[new_key.replace('gate_proj', 'up_proj')] = quant_weight_fp16(w3)
        elif 'w2' in key or 'wo' in key:
            final_result[new_key] = quant_weight_fp16(value)
        else:
            final_result[new_key] = value.clone()

    print(f"Saving checkpoint to {output_dir}/model.safetensors")
    # torch.save(final_result, checkpoint_dir / "model.pth")
    save_file(final_result, f"{output_dir}/model.safetensors", metadata={"format": "pt"})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert TorchScale checkpoint.')
    parser.add_argument('--input_file', type=Path, default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"))
    parser.add_argument('--output_dir', type=str, default="checkpoint_last.pt")
    parser.add_argument('--model_name', type=str, default="2B")

    args = parser.parse_args()
    convert_ts_checkpoint(
        input_file=args.input_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )
