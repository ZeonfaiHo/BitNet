import model as fast
from tokenizer import Tokenizer
import torch
from pathlib import Path
from sys import argv
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)

device = "cuda:0"
ckpt_dir = "/ddn/shumma/exp/sft/bitnet_2b_4t_sft_glan2_math_0306_5e-5_8k_6ep_chatformat_loss_2/updates_200000"

model_args_fp16 = fast.ModelArgs(use_kernel=False)
tokenizer = Tokenizer("./tokenizer.model")

torch.set_default_device(device)
torch.set_default_dtype(torch.bfloat16)

fp16_model = fast.Transformer(model_args_fp16)

fp16_ckpt_path = str(Path(ckpt_dir) / "model_state_fp16.pt")
fp16_checkpoint = torch.load(fp16_ckpt_path, map_location="cpu")
fp16_model.load_state_dict(fp16_checkpoint, strict=True)


model_args_int2 = fast.ModelArgs(use_kernel=True)

int2_model = fast.Transformer(model_args_int2)

int2_ckpt_path = str(Path(ckpt_dir) / "model_state_int2.pt")
int2_checkpoint = torch.load(int2_ckpt_path, map_location="cpu")
int2_model.load_state_dict(int2_checkpoint, strict=True)

ids = tokenizer.encode('What', bos=False, eos=False)
output_ids = [id for id in ids]
ids = torch.tensor(ids).cuda()
print(ids)

_cache = fast.make_cache(
    args=model_args_fp16,
    length=64,
)

seq_lens = [len(ids)]

bias = AttnBias.from_seqlens(
    q_seqlen=seq_lens,
    kv_seqlen=seq_lens,
    kv_padding=64,
)
bias.q_seqinfo.to("cuda")
bias.k_seqinfo.to("cuda")

if argv[1] == 'fp16':
    output = fp16_model.forward_with_attn_bias(
        token_values=ids,
        attn_bias=bias,
        cache=_cache,
    )
else:
    output = int2_model.forward_with_attn_bias(
        token_values=ids,
        attn_bias=bias,
        cache=_cache,
    )

print(output)