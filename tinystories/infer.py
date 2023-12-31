import math
import os
import time
from functools import partial
import torch
from dataloader import Task
from typing import Optional
from dataclasses import dataclass
from contextlib import nullcontext

from model import Transformer, ModelArgs
from simple_pre_tokenizer import Tokenizer


max_seq_len = 512
vocab_source = "custom" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 512 # the Llama 2 tokenizer has 32K tokens

# I/O
out_dir = "out"
device = "cuda"

# model
dim = 128
n_layers = 6
n_heads = 8
n_kv_heads = 4
multiple_of = 4
dropout = 0.2

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)


ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint =  torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint["model_args"]

# force these config attributes to be equal otherwise we can not even resume training
# the rest of the attributes can stay as desired from command line
for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
    model_args[k] = checkpoint_model_args[k]
# create the model
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
model = model.to(device)
state_dict = checkpoint["model"]
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        
model.load_state_dict(state_dict)

# Set the model in evaluation mode
model.eval()

DATA_CACHE_DIR = "tokenizer-data/"
vocab_size = 512
tokenizer_model = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")
tokenizer = Tokenizer(tokenizer_model)

gen_tokens = model.generate(
    idx=torch.tensor(tokenizer.encode("Lily saw a cat", bos=True, eos=False), dtype=torch.long).unsqueeze(0).to(device),
    max_new_tokens=2000)

print(tokenizer.decode(gen_tokens[0].detach().cpu().numpy().tolist()))