import torch
import torch.nn.functional as F
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout_rate):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout_rate):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(head_size, n_embd, block_size, dropout_rate)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Block: communication followed by computation"""

    def __init__(self, n_embd, n_head, block_size, dropout_rate, use_flash=True):
        super().__init__()
        head_size = n_embd // n_head
        if use_flash:
            self.sa = FlashMultiHeadAttention(
                n_head, head_size, n_embd, block_size, dropout_rate
            )
        else:
            self.sa = MultiHeadAttention(
                n_head, head_size, n_embd, block_size, dropout_rate
            )

        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class FlashMultiHeadAttention(nn.Module):
    """Combines Single Attentino and Multi Head attention in one block"""

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout_rate):
        super().__init__()
        assert n_embd % num_heads == 0

        self.c_attn = nn.Linear(
            n_embd, 3 * n_embd, bias=False
        )  # This layer is equivalent to computing k,q,v in Attention above
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # regularization
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)
        self.n_head = num_heads
        self.n_embd = n_embd
        self.dropout_rate = dropout_rate
        self.head_size = head_size

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if self.flash:
            print("self attention would be used")

    def forward(self, x, is_training=True):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k, q, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)

        y = None
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_rate if is_training else 0,
                is_causal=True,
            )
            y = (
                y.transpose(1, 2).contiguous().view(B, T, C)
            )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class GptModel(nn.Module):
    def __init__(
        self, vocab_size, num_heads, n_embd, block_size, device, dropout_rate, n_layer
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=n_embd,
                    n_head=num_heads,
                    block_size=block_size,
                    dropout_rate=dropout_rate,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (B, T, C)
        x = tok_emb + pos_emb  # ( B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the prediction
            logits, loss = self(idx[:, -self.block_size :])
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sample index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
