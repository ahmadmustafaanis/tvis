"""Small classifier with switchable, classic training bugs. Fixture for smoke.sh: each --bug value
must make `tvis_debug.py health` report the matching check. Plain PyTorch; knows nothing about tvis."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Blobs(Dataset):
    classes = ["a", "b", "c", "d"]

    def __init__(self, n: int, raw_pixels: bool):
        g = torch.Generator().manual_seed(0)
        self.y = torch.randint(4, (n,), generator=g)
        centers = torch.randn(4, 1, 16, 16, generator=g)
        self.x = centers[self.y] + 0.5 * torch.randn(n, 1, 16, 16, generator=g)
        if raw_pixels:  # forgot to normalise: 0..255 pixel values
            self.x = ((self.x - self.x.min()) / (self.x.max() - self.x.min()) * 255).round()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.x[i], int(self.y[i])


class Net(nn.Module):
    def __init__(self, dead_bias: bool):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.fc1 = nn.Linear(8 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 4)
        if dead_bias:
            nn.init.constant_(self.fc1.bias, -50.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bug", default="none", choices=["none", "softmax", "nozero", "unnorm", "frozen", "dead", "lr"]
    )
    args = p.parse_args()
    torch.manual_seed(0)
    loader = DataLoader(Blobs(64, raw_pixels=args.bug == "unnorm"), batch_size=16, shuffle=True)
    model = Net(dead_bias=args.bug == "dead")
    params = model.parameters() if args.bug != "frozen" else model.fc2.parameters()
    opt = torch.optim.SGD(params, lr=50.0 if args.bug == "lr" else 0.05)
    for _ in range(3):
        for x, y in loader:
            logits = model(x)
            if args.bug == "softmax":
                logits = logits.softmax(dim=-1)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            if args.bug != "nozero":
                opt.zero_grad()


if __name__ == "__main__":
    main()
