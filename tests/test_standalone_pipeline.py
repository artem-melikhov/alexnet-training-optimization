import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import load_training_data, make_loaders
from utils.profiling import capture
from utils.train_utils import Metrics, fit


def synthetic_data(path):
    train = path / "train"
    train.mkdir()
    rng = np.random.default_rng(52)
    ids = list(range(1, 25))
    for index in ids:
        image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(image).save(train / f"{index}.png")
    pd.DataFrame({"id": ids, "label": ["cat", "dog"] * 12}).to_csv(
        path / "trainLabels.csv", index=False,
    )
    return load_training_data(path, seed=52)


def test_batch_preprocessing_matches_original_scale(tmp_path):
    data = synthetic_data(tmp_path)
    train, val, mean, std = data
    source = train.dataset.tensors[0]
    assert source.dtype == torch.uint8
    assert set(train.indices).isdisjoint(val.indices)
    train_loader, val_loader, preprocess = make_loaders(
        data, torch.device("cpu"), batch_size=3, seed=52, workers=0, augment=True,
    )
    x, y = next(iter(train_loader))
    assert x.dtype == torch.uint8 and y.dtype == torch.long
    original_x, _ = next(iter(val_loader))
    old_train = source[train.indices].float() / 255
    old_mean = old_train.mean((0, 2, 3)).reshape(1, -1, 1, 1)
    old_std = old_train.std((0, 2, 3), correction=0).reshape(1, -1, 1, 1)
    expected = (original_x.float() / 255 - old_mean) / old_std
    torch.testing.assert_close(preprocess(original_x), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(mean, source[train.indices].float().mean((0, 2, 3)))
    torch.testing.assert_close(std, source[train.indices].float().std((0, 2, 3), correction=0))


def test_metrics_weight_batches_and_detach():
    metrics = Metrics(torch.device("cpu"))
    metrics.update(
        torch.tensor(0.5, requires_grad=True),
        torch.tensor([[2.0, 1.0], [2.0, 1.0]], requires_grad=True),
        torch.tensor([0, 1]),
    )
    metrics.update(torch.tensor(2.0, requires_grad=True), torch.tensor([[0.0, 2.0]]), torch.tensor([1]))
    assert metrics.values() == (1.0, 2 / 3)
    assert not metrics.loss.requires_grad


def test_training_and_profile_write_artifacts(tmp_path, monkeypatch):
    import utils.train_utils as training

    data = synthetic_data(tmp_path)
    # Small model here keeps the integration test quick while exercising the
    # same image pipeline, training step and profiler used by both notebooks.
    monkeypatch.setattr(training, "AlexNet", lambda: nn.Sequential(nn.Flatten(), nn.Linear(3072, 10)))
    device = torch.device("cpu")
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        history = fit(
            data, device, tmp_path / "fp32", epochs=1, seed=52, batch_size=2, workers=0,
        )
        assert len(history) == 1
        assert history["val_acc"].between(0, 1).all()
        assert (tmp_path / "fp32" / "best.pth").is_file()
        saved = pd.read_csv(tmp_path / "fp32" / "history.csv")
        pd.testing.assert_frame_equal(history, saved)
        trace = tmp_path / "trace.json"
        capture(data, device, trace, seed=52, batch_size=2, workers=0)
        events = json.loads(trace.read_text())["traceEvents"]
        names = [event["name"] for event in events]
        assert names.count("01_data_loading") == 5
        assert names.count("02b_standardize_batch") == 5
        assert names.count("06_metrics") == 5
    finally:
        torch.set_num_threads(prev_threads)



def test_invalid_epoch_count_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="не меньше 1"):
        fit(None, torch.device("cpu"), tmp_path, epochs=0, seed=52, batch_size=2, workers=0)
