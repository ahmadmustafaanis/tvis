"""Tiny word-level transformer language model with a hand-written attention function.

Builds a HuggingFace tokenizer locally (no download). Plain PyTorch; knows nothing about tvis.

    python examples/text_lm/train.py
    tvis run --steps 2 examples/text_lm/train.py
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

SENTENCES = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "a bird flew over the house",
    "the cat chased the dog",
    "a dog chased a cat over the mat",
    "the bird sat on the house",
    "a cat sat on a log",
    "the dog flew over the mat",
]


def build_tokenizer() -> PreTrainedTokenizerFast:
    words = sorted({w for s in SENTENCES for w in s.split()})
    vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3} | {w: i + 4 for i, w in enumerate(words)}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=core, pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]"
    )


class SentenceDataset(Dataset):
    def __init__(self, sentences: list[str]):
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> str:
        return self.sentences[index]


def make_collate(tokenizer: PreTrainedTokenizerFast):
    def collate(texts: list[str]) -> dict[str, torch.Tensor]:
        enc = tokenizer([f"[BOS] {t} [EOS]" for t in texts], padding=True, return_tensors="pt")
        ids = enc["input_ids"]
        labels = ids[:, 1:].clone()
        labels[enc["attention_mask"][:, 1:] == 0] = -100
        return {"input_ids": ids[:, :-1], "labels": labels}

    return collate


def multi_head_new_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_heads: int) -> torch.Tensor:
    b, t, d = q.shape
    q = q.view(b, t, n_heads, d // n_heads).transpose(1, 2)
    k = k.view(b, t, n_heads, d // n_heads).transpose(1, 2)
    v = v.view(b, t, n_heads, d // n_heads).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d // n_heads)
    causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    weights = scores.softmax(dim=-1)
    out = weights @ v
    return out.transpose(1, 2).reshape(b, t, d)


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        x = x + self.proj(multi_head_new_attention(q, k, v, self.n_heads))
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d: int = 32, n_heads: int = 4, n_layers: int = 2, max_len: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.embed(input_ids) + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    tokenizer = build_tokenizer()
    loader = DataLoader(
        SentenceDataset(SENTENCES),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer),
    )
    model = TinyLM(len(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for _epoch in range(args.epochs):
        for batch in loader:
            logits = model(batch["input_ids"])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1), ignore_index=-100
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.log:
                with open(args.log, "a") as log:
                    log.write(json.dumps({"loss": loss.item()}) + "\n")
    print("done")


if __name__ == "__main__":
    main()
