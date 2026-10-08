"""Datasets used as "user code" by the data-pipeline tests."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, IterableDataset

LABELS = ["cat", "dog", "bird"]


class VectorDataset(Dataset):
    classes = LABELS

    def __init__(self, n: int = 16, d: int = 6):
        gen = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, d, generator=gen)
        self.y = torch.arange(n) % len(LABELS)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], int(self.y[index])


class BatchedFetchDataset(VectorDataset):
    """Implements __getitems__ (batched fetching); tvis must still see per-sample items."""

    def __getitems__(self, indices):
        return [self[i] for i in indices]


class StreamDataset(IterableDataset):
    def __iter__(self):
        gen = torch.Generator().manual_seed(0)
        for i in range(12):
            yield torch.randn(6, generator=gen), i % 3


class ImageDataset(Dataset):
    classes = LABELS

    def __init__(self, transform, n: int = 8):
        self.transform, self.n = transform, n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        from PIL import Image

        image = Image.new("RGB", (12, 10), (index * 20 % 256, 100, 200))
        return self.transform(image), index % len(LABELS)


class TextDataset(Dataset):
    SENTENCES = ["the cat sat", "a dog ran far away", "birds fly", "the dog sat on a mat"]

    def __len__(self) -> int:
        return len(self.SENTENCES)

    def __getitem__(self, index: int) -> str:
        return self.SENTENCES[index]


def build_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = sorted({w for s in TextDataset.SENTENCES for w in s.split()})
    vocab = {"[PAD]": 0, "[UNK]": 1} | {w: i + 2 for i, w in enumerate(words)}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=core, pad_token="[PAD]", unk_token="[UNK]")


def text_collate(tokenizer):
    def collate(texts):
        enc = tokenizer(list(texts), padding=True, return_tensors="pt")
        ids = enc["input_ids"]
        labels = ids.clone()
        labels[enc["attention_mask"] == 0] = -100
        return ids, labels

    return collate
