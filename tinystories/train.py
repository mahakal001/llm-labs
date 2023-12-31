import math
import os
import time
from functools import partial
import torch
from dataloader import Task
from typing import Optional


@dataclass
class ModelArgs:
    # default hyperparameters for the tinystories
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = 4096
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 256
    dropout: float = 0.0

batch_size=32
max_seq_len=64
vocab_size=32
vocab_size=4096
device="cuda"
gradient_accumulation_steps = 1

# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    device=device,
    num_workers=0,
)

master_process = True
seed_offset = 0
ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len


