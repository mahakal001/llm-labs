from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dropout: float = 0.2
    block_size: int = 256
    batch_size: int = 64
    num_heads: int = 6
    n_embd: int = 384
    lr: float = 3e-4
    max_iters: int = 10000
    eval_iters: int = 200
    eval_interval: int = 500
    n_layer: int = 6
