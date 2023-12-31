# Taken directly from karpathy's llama2.c with some modifications to make it simple.

import os
import struct
import argparse
from typing import List
import glob
from sentencepiece import SentencePieceProcessor
import json
from tqdm import tqdm
import numpy as np

DATA_CACHE_DIR = "tokenizer-data"

class Tokenizer:
    def __init__(self, tokenizer_model=None):

        model_path = tokenizer_model
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        #print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t
    
    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

def tokenize(vocab_size):
    
    tokenizer_model = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")
    enc = Tokenizer(tokenizer_model)

    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join("/home/mkal001/personal/mygithub/llm-labs/dataset", "tinystories-data")
    filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    for shard in tqdm(filenames):
        with open(shard, "r") as f:
            data = json.load(f)

            all_tokens = []
            for example in tqdm(data):
                text = example["story"]
                text = text.strip()  # get rid of leading/trailing whitespace
                tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
                all_tokens.extend(tokens)

            # convert to uint16 nparray
            all_tokens = np.array(all_tokens, dtype=np.uint16)

       
            # save .bin files into a new tok{N} directory
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
            shard_basename = os.path.basename(shard)
            bin_basename = shard_basename.replace(".json", ".bin")
            tokenized_filename = os.path.join(bin_dir, bin_basename)
            # write the bytes
            with open(tokenized_filename, "wb") as f:
                f.write(all_tokens.tobytes())

            # calculate the average sequence length (they are separated by BOS=1)
            avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
            print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")

tokenize(vocab_size=4096)