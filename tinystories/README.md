# Everthing here is copied from Andrej Karpathy's Llama.c. Some code is removed here and there.
# Steps
1. Train a custom tokenizer specific to our dataset.
2. It makes sense to tokenize the whole dataset at once using this trained tokenizer, but not necessary. Can be done on fly.
3. Use SILU as activation and  RoPE embeddings as positional embedding, RMSNorm replacing LayerNorm.
4. Train with weight decay and lr decay.