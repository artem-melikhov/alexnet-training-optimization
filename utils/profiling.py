from pathlib import Path
from time import perf_counter

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from .train_utils import Metrics, setup_run, train_batch


def capture(data, device, trace_path, *, seed, batch_size, workers, augment=True, use_amp=False):
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    model, opt, criterion, scaler, loader, _, preprocess = setup_run(
        data, device, seed=seed, batch_size=batch_size, workers=workers,
        augment=augment, use_amp=use_amp,
    )
    if len(loader) < 9:
        raise ValueError("Для профиля нужно не менее 9 батчей.")
    model.train()
    metrics = Metrics(device)

    def export(prof):
        sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=25))
        prof.export_chrome_trace(str(trace_path))
        print(f"Трейс сохранён в {trace_path}")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    batches = iter(loader)
    try:
        with profile(
            activities=activities,
            schedule=schedule(wait=2, warmup=2, active=5, repeat=1),
            on_trace_ready=export,
        ) as prof:
            for _ in range(9):
                with record_function("01_data_loading"):
                    batch = next(batches)
                train_batch(batch, model, opt, criterion, device, preprocess, metrics,
                            use_amp=use_amp, scaler=scaler, mark=record_function)
                prof.step()
    finally:
        del batches, loader
    loss, acc = metrics.values()
    print(f"Loss за 9 батчей {loss:.4f}, accuracy {acc:.2%}")


def benchmark_workers(data, device, *, seed, batch_size, workers):
    """Measure 30 steps after 10 warmup steps with a fresh model."""
    model, opt, criterion, scaler, loader, _, preprocess = setup_run(
        data, device, seed=seed, batch_size=batch_size, workers=workers,
        augment=True, use_amp=False,
    )
    if len(loader) < 40:
        raise ValueError("Для замера нужно не менее 40 батчей.")
    model.train()
    metrics = Metrics(device)
    batches = iter(loader)
    try:
        for step in range(40):
            if step == 10:
                torch.cuda.synchronize(device)
                start = perf_counter()
            train_batch(next(batches), model, opt, criterion, device, preprocess, metrics,
                        scaler=scaler)
        torch.cuda.synchronize(device)
        return (perf_counter() - start) * 1000 / 30
    finally:
        del batches, loader
