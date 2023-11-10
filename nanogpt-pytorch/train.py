import torch
from dataloader import train_data, val_data, vocab_size, decode
import numpy as np
from configs import ModelArgs
from gpt import GptModel

config = ModelArgs()
block_size = config.block_size
batch_size = config.batch_size

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )

    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


xb, yb = get_batch("train")
model = GptModel(
    vocab_size=vocab_size,
    num_heads=config.num_heads,
    n_embd=config.n_embd,
    block_size=config.block_size,
    device=device,
    dropout_rate=config.dropout,
    n_layer=config.n_layer,
)

model = model.to(device)
print("compiling for max-tune")
model = torch.compile(model, mode="max-autotune")
print("compilation done")


logits, loss = model(xb, yb)
print(logits.shape)
print(loss)
print(
    decode(
        model.generate(
            idx=torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=200
        )[0]
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )
)

learning_rate = 1e-3
# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
)
print(f"Total trainable params:- {trainable_params}")

for iter in range(config.max_iters):
    if iter % config.eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model_trained.pt")
print(
    decode(
        model.generate(
            idx=torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=1000
        )[0]
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )
)
