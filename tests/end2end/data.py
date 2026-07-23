"""MNIST helpers for end-to-end tests (no torchvision dependency)."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

_MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
}


class MNIST(Dataset):
    mean = 0.1307
    std = 0.3081

    def __init__(self, root: Path):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        images_path = self._ensure(root, "train-images-idx3-ubyte.gz")
        labels_path = self._ensure(root, "train-labels-idx1-ubyte.gz")
        self.images = self._read_images(images_path)
        self.labels = self._read_labels(labels_path)

    @staticmethod
    def _ensure(root: Path, name: str) -> Path:
        path = root / name
        if not path.exists():
            urllib.request.urlretrieve(_MNIST_URLS[name], path)
        return path

    @staticmethod
    def _read_images(path: Path) -> torch.Tensor:
        with gzip.open(path, "rb") as handle:
            magic, n, rows, cols = struct.unpack(">IIII", handle.read(16))
            if magic != 2051:
                raise ValueError(f"Unexpected MNIST image magic: {magic}")
            data = torch.frombuffer(bytearray(handle.read()), dtype=torch.uint8)
            return data.reshape(n, 1, rows, cols).float().div_(255.0)

    @staticmethod
    def _read_labels(path: Path) -> torch.Tensor:
        with gzip.open(path, "rb") as handle:
            magic, n = struct.unpack(">II", handle.read(8))
            if magic != 2049:
                raise ValueError(f"Unexpected MNIST label magic: {magic}")
            return torch.frombuffer(bytearray(handle.read()), dtype=torch.uint8).long()

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int):
        image = (self.images[index] - self.mean) / self.std
        return image, self.labels[index]


def mnist_dataloader(
    root: Path,
    *,
    subset_size: int = 1024,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Return a short MNIST training loader suitable for e2e smoke tests."""
    dataset = MNIST(root)
    return DataLoader(
        Subset(dataset, range(min(subset_size, len(dataset)))),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
