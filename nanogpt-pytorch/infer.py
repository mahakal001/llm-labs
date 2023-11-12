import torch
import numpy as np
from configs import ModelArgs
from gpt import GptModel
from dataloader import vocab_size, decode

config = ModelArgs()
block_size = config.block_size
batch_size = config.batch_size

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

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
model = torch.compile(model, mode="max-autotune")

# Load the saved model dictionary
saved_model_dict = torch.load('model_trained.pt')
# Update the model with the saved weights
model.load_state_dict(saved_model_dict)


# Set the model in evaluation mode
model.eval()

print(
    decode(
        model.generate(
            idx=torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=10000
        )[0]
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )
)

