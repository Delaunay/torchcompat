"""End-to-end training with torchcompat.

Shows the portable training loop. The same code targets CUDA, ROCm, XPU,
Gaudi, Tenstorrent, XLA, or CPU — only ``torchcompat.core`` device helpers
change vs a CUDA-only script::

    # CUDA-only                         # torchcompat
    device = torch.device("cuda:0")      device = accelerator.fetch_device(0)
    with torch.amp.autocast("cuda"):     with accelerator.amp.autocast():
    optimizer.step()                    accelerator.optimizer_step(optimizer)
    torch.cuda.synchronize()            accelerator.synchronize()
"""

from __future__ import annotations

import os
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchcompat.core as accelerator
from tests.end2end.data import mnist_dataloader

# ``dp`` | ``tp`` | ``none`` — mesh parallelism (TT/XLA, needs >= 2 devices).
PARALLELISM = os.environ.get("TORCHCOMPAT_PARALLELISM", "dp").lower()


class LeNet(nn.Module):
    """Classic LeNet-5 for 28x28 MNIST."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def setup_mesh(parallelism: str = PARALLELISM) -> None:
    """Configure SPMD mesh for data-parallel or tensor-parallel training.

    Uses ``init_mesh_group`` (TT-only; no-op on other backends). Must run
    before ``manual_seed`` / ``fetch_device`` / any tensor creation.
    """
    if parallelism in ("", "none", "off"):
        return

    # Prefer sysfs / TT_VISIBLE_DEVICES count (does not init XLA without SPMD).
    num_devices = accelerator.device_count()
    if num_devices < 2:
        return

    if parallelism == "dp":
        # Shard the batch dimension across all devices.
        accelerator.init_mesh_group(
            mesh_shape=(num_devices,),
            mesh_axis_names=("batch",),
            input_sharding_dim_arg="batch",
        )
        return

    if parallelism == "tp":
        # Alternating column/row TP on the classifier (dims divisible by 2).
        accelerator.init_mesh_group(
            mesh_shape=(1, num_devices),
            mesh_axis_names=("batch", "model"),
            input_sharding_dim_arg=None,
            model_sharding_patterns_arg=[
                (r"fc1$", ("model", None)),
                (r"fc2$", (None, "model")),
                (r"fc3$", (None, "model")),
            ],
        )
        return

    raise ValueError(
        f"Unknown PARALLELISM={parallelism!r}; expected 'dp', 'tp', or 'none'"
    )


def test_training(tmp_path):
    # Mesh / SPMD first — before seeds, devices, or model weights on XLA.
    setup_mesh(PARALLELISM)

    accelerator.manual_seed(0)
    device = accelerator.fetch_device(0)

    loader = mnist_dataloader(tmp_path / "mnist")
    model = LeNet().to(device)
    model = accelerator.shard_model(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    elasped = []
    perf = []

    for _epoch in range(3):
        samples = 0
        start = accelerator.Event()
        start.record()

        for i, (x, y) in enumerate(loader):
            print(f"{_epoch}, {i}")

            samples += x.shape[0]
            x, y = accelerator.prepare_batch(x, y)

            with accelerator.amp.autocast():
                pred = model(x)
                loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            accelerator.optimizer_step(optimizer)

            losses.append(loss.detach())

        end = accelerator.Event()
        end.record()
        # Event.elapsed_time returns milliseconds.
        e = start.elapsed_time(end) / 1000.0

        elasped.append(e)
        perf.append(samples / e if e > 0 else 0.0)

    print(elasped)
    print(perf)

    accelerator.synchronize()
    avg = [l.item() for l in losses]
    print(sum(avg) / len(avg))


if __name__ == "__main__":
    test_training(Path("/tmp"))
