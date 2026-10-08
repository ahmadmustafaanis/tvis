"""Toy models and training loops used as "user code" by the capture tests.

This directory is the project root in these tests, so every function here is traced exactly as a
user's own code would be. Line-number assertions locate statements via `line_of`, never hard-coded.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from torch import nn


def scale_and_shift(x: torch.Tensor, scale: float, shift: torch.Tensor) -> torch.Tensor:
    y = x * scale
    return y + shift


class ToyMLP(nn.Module):
    def __init__(self, d_in: int = 6, d_hidden: int = 8, n_classes: int = 3, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, n_classes)
        self.shift = nn.Parameter(torch.zeros(d_hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = scale_and_shift(h, 2.0, self.shift)
        return self.fc2(self.drop(h))


def attention_like(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1)
    weights = scores.softmax(dim=-1)
    return weights.sum(dim=1)


class AttnModel(nn.Module):
    def __init__(self, d: int = 4):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return attention_like(self.q(x), self.k(x))


class BatchNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.fc(x))


def make_batches(n: int, batch_size: int = 4, d_in: int = 6, n_classes: int = 3, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(batch_size, d_in, generator=gen), torch.randint(n_classes, (batch_size,), generator=gen))
        for _ in range(n)
    ]


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def train(
    model: nn.Module, optimizer: torch.optim.Optimizer, batches, accum: int = 1, losses: list | None = None
):
    """A standard loop: forward, loss via a project function, backward, (accumulated) step."""
    for i, (x, y) in enumerate(batches):
        loss = compute_loss(model(x), y) / accum
        if losses is not None:
            losses.append(loss.detach().clone())
        loss.backward()
        if (i + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad()


def train_manual_sgd(model: nn.Module, batches, lr: float = 0.1):
    """No torch.optim: parameters are updated by hand."""
    for x, y in batches:
        loss = compute_loss(model(x), y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
                p.grad = None


def line_of(func, needle: str) -> int:
    """Absolute line number of the first source line of `func` containing `needle`."""
    lines, start = inspect.getsourcelines(func)
    for offset, text in enumerate(lines):
        if needle in text:
            return start + offset
    raise AssertionError(f"{needle!r} not found in {func.__name__}")
