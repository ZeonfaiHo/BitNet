# convert_safetensors_to_pth.py

import re
import torch
from pathlib import Path
from safetensors.torch import load_file
from einops import rearrange
from dataclasses import dataclass
from typing import Optional

# 和正向脚本里一样的模型配置
transformer_configs = {
    "2B": dict(n_layer=30, n_head=20, dim=2560, vocab_size=128256, n_local_heads=5, intermediate_size=6912),
}

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
            # 保证可被 256 整除
            self.intermediate_size = n_hidden + (256 - n_hidden % 256) if n_hidden % 256 else n_hidden
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        config = [k for k in transformer_configs if k in name.upper() or k in name]
        assert len(config) == 1, f"Unknown model name: {name}"
        return cls(**transformer_configs[config[0]])

def invert_convert_q(w: torch.Tensor, config: ModelArgs) -> torch.Tensor:
    # 原来的 convert_q: (h*d*l, i) -> rearrange '(h d l) i -> (h l d) i'
    # 逆操作：'(h l d) i -> (h d l) i'
    return rearrange(w, '(h l d) i -> (h d l) i', h=config.n_head, l=2)

def invert_convert_k(w: torch.Tensor, config: ModelArgs) -> torch.Tensor:
    # 同理
    return rearrange(w, '(h l d) i -> (h d l) i', h=config.n_local_heads, l=2)

def convert_back(
    safetensors_path: str,
    output_pth: str,
    model_name: Optional[str] = None,
):
    # 加载 safetensors
    st_dict = load_file(safetensors_path)

    cfg = ModelArgs.from_name(model_name)
    print(f"使用模型配置: {cfg}")

    recovered: dict = {}

    for layer in range(cfg.n_layer):
        # 注意字符串格式要和正向脚本中的一致
        base = f"model.layers.{layer}."

        # 1) q/k/v 合并回 wqkv
        wq = st_dict[f"{base}self_attn.q_proj.weight"]
        wk = st_dict[f"{base}self_attn.k_proj.weight"]
        wv = st_dict[f"{base}self_attn.v_proj.weight"]

        # 逆向 rearrange
        wq = invert_convert_q(wq, cfg)
        wk = invert_convert_k(wk, cfg)
        # wv = invert_convert_q(wv, cfg)  # v_proj 用全局 heads

        # concat 回原来的 [3*dim, dim]
        wqkv = torch.cat([wq, wk, wv], dim=0)
        recovered[f"layers.{layer}.attention.wqkv.weight"] = wqkv

        # 2) wo 直接映射回去
        recovered[f"layers.{layer}.attention.wo.weight"] = st_dict[f"{base}self_attn.o_proj.weight"]

        # 3) attention_norm & ffn_norm & 子 norm
        recovered[f"layers.{layer}.attention_norm.weight"] = st_dict[f"{base}input_layernorm.weight"]
        recovered[f"layers.{layer}.ffn_norm.weight"] = st_dict[f"{base}post_attention_layernorm.weight"]
        recovered[f"layers.{layer}.attention.attn_sub_norm.weight"] = st_dict[f"{base}self_attn.attn_sub_norm.weight"]
        recovered[f"layers.{layer}.feed_forward.ffn_sub_norm.weight"] = st_dict[f"{base}mlp.ffn_sub_norm.weight"]

        # 4) Feed-forward 的 w13 (gate + up) 和 w2 (down)
        gate = st_dict[f"{base}mlp.gate_proj.weight"]
        up   = st_dict[f"{base}mlp.up_proj.weight"]
        w13  = torch.cat([gate, up], dim=0)
        recovered[f"layers.{layer}.feed_forward.w13.weight"] = w13

        recovered[f"layers.{layer}.feed_forward.w2.weight"] = st_dict[f"{base}mlp.down_proj.weight"]

    # 嵌入、输出和最终 norm
    recovered["tok_embeddings.weight"] = st_dict["model.embed_tokens.weight"]
    recovered["output.weight"]         = st_dict["model.embed_tokens.weight"]
    recovered["norm.weight"]           = st_dict["model.norm.weight"]

    # 保存 .pth
    print(f"保存到 {output_pth}")
    torch.save(recovered, output_pth)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="把 Safetensors 转回 Torch .pth 检查点")
    parser.add_argument(
        "--safetensors_file", type=str, required=True,
        help="输入的 .safetensors 文件路径"
    )
    parser.add_argument(
        "--output_pth", type=str, default="model_converted.pth",
        help="输出的 .pth 文件路径"
    )
    parser.add_argument(
        "--model_name", type=str, default="2B",
        help="使用的模型配置名称（如 2B）"
    )
    args = parser.parse_args()

    convert_back(
        safetensors_path=args.safetensors_file,
        output_pth=args.output_pth,
        model_name=args.model_name,
    )
