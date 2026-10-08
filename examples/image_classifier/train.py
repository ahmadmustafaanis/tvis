"""Tiny image classifier on synthetic shapes. Plain PyTorch; knows nothing about tvis.

python examples/image_classifier/train.py --epochs 1
tvis run --steps 3 examples/image_classifier/train.py --accum 2
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ["circle", "square", "triangle"]


class ShapesDataset(Dataset):
    """Draws a random shape per index (deterministic per index) as a PIL image."""

    classes = CLASSES

    def __init__(self, n: int, transform=None, size: int = 40):
        self.n, self.transform, self.size = n, transform, size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        gen = torch.Generator().manual_seed(index)
        label = int(torch.randint(len(CLASSES), (1,), generator=gen))
        image = draw_shape(label, self.size, gen)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def draw_shape(label: int, size: int, gen: torch.Generator) -> Image.Image:
    image = Image.new("RGB", (size, size), tuple(int(v) for v in torch.randint(0, 60, (3,), generator=gen)))
    draw = ImageDraw.Draw(image)
    color = tuple(int(v) for v in torch.randint(150, 256, (3,), generator=gen))
    lo, hi = sorted(int(v) for v in torch.randint(4, size - 4, (2,), generator=gen))
    hi = max(hi, lo + 8)
    if CLASSES[label] == "circle":
        draw.ellipse([lo, lo, hi, hi], fill=color)
    elif CLASSES[label] == "square":
        draw.rectangle([lo, lo, hi, hi], fill=color)
    else:
        draw.polygon([(lo, hi), (hi, hi), ((lo + hi) // 2, lo)], fill=color)
    return image


def conv_block(x: torch.Tensor, conv: nn.Conv2d, norm: nn.BatchNorm2d) -> torch.Tensor:
    x = conv(x)
    x = norm(x)
    x = F.gelu(x)
    return F.max_pool2d(x, 2)


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1, self.norm1 = nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16)
        self.conv2, self.norm2 = nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.1), nn.Linear(32 * 8 * 8, n_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = conv_block(images, self.conv1, self.norm1)
        x = conv_block(x, self.conv2, self.norm2)
        return self.head(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=1, help="gradient accumulation micro-batches")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--log", type=str, default=None, help="write per-step losses as JSON lines")
    args = parser.parse_args()

    torch.manual_seed(0)
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ]
    )
    loader = DataLoader(
        ShapesDataset(args.n, transform), batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    model = SmallCNN(len(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    for _epoch in range(args.epochs):
        for i, (images, labels) in enumerate(loader):
            logits = model(images)
            loss = F.cross_entropy(logits, labels) / args.accum
            loss.backward()
            if args.log:
                with open(args.log, "a") as log:
                    log.write(json.dumps({"batch": i, "loss": loss.item()}) + "\n")
            if (i + 1) % args.accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
    print(f"done: {step} steps")


if __name__ == "__main__":
    main()
