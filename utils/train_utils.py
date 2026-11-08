from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from torch import nn

from .alexnet import AlexNet
from .data_utils import make_loaders


class Metrics:
    def __init__(self, device):
        self.loss = torch.zeros((), device=device)
        self.correct = torch.zeros((), dtype=torch.int64, device=device)
        self.samples = 0

    def update(self, loss, pred, y):
        count = y.size(0)
        self.loss += loss.detach() * count
        self.correct += (pred.detach().argmax(dim=1) == y).sum()
        self.samples += count

    def values(self):
        return self.loss.item() / self.samples, self.correct.item() / self.samples


def prepare_batch(batch, device, preprocess, mark):
    x, y = batch
    with mark("02_to_device"):
        x, y = x.to(device), y.to(device)
    if preprocess is not None:
        with mark("02b_standardize_batch"):
            x = preprocess(x)
    return x, y


def train_batch(batch, model, optimizer, criterion, device, preprocess, metrics,
                *, use_amp=False, scaler=None, mark=nullcontext):
    x, y = prepare_batch(batch, device, preprocess, mark)
    with mark("03_forward"):
        pred = model(x)
        loss = criterion(pred, y)
    with mark("04_backward"):
        optimizer.zero_grad()
        loss.backward()
    with mark("05_optimizer"):
        optimizer.step()
    with mark("06_metrics"):
        metrics.update(loss, pred, y)


def setup_run(data, device, *, seed, batch_size, workers, augment, use_amp):
    torch.manual_seed(seed)
    model = AlexNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = None
    train_loader, val_loader, preprocess = make_loaders(
        data, device, batch_size=batch_size, seed=seed, workers=workers, augment=augment,
    )
    return model, optimizer, criterion, scaler, train_loader, val_loader, preprocess


def fit(data, device, output_dir, *, epochs, seed, batch_size, workers, use_amp=False):
    """Run one fresh model and persist validation history after every epoch."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, opt, criterion, scaler, train_loader, val_loader, preprocess = setup_run(
        data, device, seed=seed, batch_size=batch_size, workers=workers,
        augment=True, use_amp=use_amp,
    )
    history = []
    best = -1.0
    mode = "AMP FP16" if use_amp else "FP32"
    for epoch in range(1, epochs + 1):
        start = perf_counter()
        model.train()
        train_metrics = Metrics(device)
        for batch in train_loader:
            train_batch(batch, model, opt, criterion, device, preprocess, train_metrics,
                        use_amp=use_amp, scaler=scaler)
        train_loss, train_acc = train_metrics.values()

        model.eval()
        val_metrics = Metrics(device)
        with torch.inference_mode():
            for batch in val_loader:
                x, y = prepare_batch(batch, device, preprocess, nullcontext)
                pred = model(x)
                val_metrics.update(criterion(pred, y), pred, y)
        val_loss, val_acc = val_metrics.values()
        seconds = perf_counter() - start
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, seconds=seconds))
        frame = pd.DataFrame(history)
        frame.to_csv(output_dir / "history.csv", index=False)
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), output_dir / "best.pth")
        print(f"{mode} | эпоха {epoch:3d}/{epochs} | train {train_acc:.2%} | "
              f"val {val_acc:.2%} | val loss {val_loss:.4f} | {seconds:.1f} с")

    return frame
