
import torch
from PIL import Image
from torchvision.transforms import PILToTensor, v2
import os
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from torch.utils.data import Subset


class AutoAugumentationDataset(Dataset):
    """Применяет AutoAugment для CIFAR-10 при каждой выдаче изображения.
    При каждом обращении к изображению выбирается случайный вариант аугментации,
    а метка класса остаётся прежней.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.augment = v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[index]

        # Загрузчик уже выдаёт uint8 [0, 255]; смена типа здесь не нужна.
        return self.augment(image), label


def calculate_mean_std(ds_train: Subset) -> tuple[torch.Tensor, torch.Tensor]:
    """Считает статистики исходного train в шкале 0–255 небольшими батчами."""
    if len(ds_train) == 0:
        raise ValueError("Нельзя вычислить статистики пустой выборки")

    # Не создаём float-копию всего train и не меняем глобальный RNG.
    loader = DataLoader(
        ds_train, batch_size=512, shuffle=False,
        generator=torch.Generator().manual_seed(0),
    )
    channel_sum = channel_square_sum = None
    pixel_count = 0
    for images, _ in loader:
        values = images.to(torch.float64)
        batch_sum = values.sum(dim=(0, 2, 3))
        batch_square_sum = values.square().sum(dim=(0, 2, 3))
        if channel_sum is None:
            channel_sum = batch_sum
            channel_square_sum = batch_square_sum
        else:
            channel_sum += batch_sum
            channel_square_sum += batch_square_sum
        pixel_count += values.shape[0] * values.shape[2] * values.shape[3]

    mean = channel_sum / pixel_count
    variance = (channel_square_sum / pixel_count - mean.square()).clamp_min(0)
    return mean.float(), variance.sqrt().float()


class BatchStandardizer:
    """Переводит uint8-батч в float32 и стандартизует на устройстве модели.

    Mean и std задаются в шкале исходных uint8-пикселей (0–255).
    Коэффициенты переносятся на устройство один раз при создании объекта.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, device):
        self.mean = mean.to(device=device, dtype=torch.float32).reshape(1, -1, 1, 1)
        self.std = std.to(device=device, dtype=torch.float32).reshape(1, -1, 1, 1)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        # float() создаёт новый тензор для uint8-входа: исходные данные не меняются.
        return (batch.float() - self.mean) / self.std


def images_to_tensor(
    images_path: str,
    csv_path: str,
    channels: int = 3,
    has_labels: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Загружает изображения в uint8 [0, 255], без деления на 255."""

    # читаем метки правильных классов для фото они сопоставлены с ними по image_id, 
    # который указан в названии файла картинки(для случая когда эти метки есть, то есть train), 
    # иначе читаем id которые идут в файле sample_submission, 
    # чтобы дать предсказания для картинок в том же порядке
    df = pd.read_csv(csv_path)
    to_tensor = PILToTensor()
    img_tensors = []

    for image_id in df["id"]:
        img_path = os.path.join(images_path, f"{image_id}.png")

        with Image.open(img_path) as img:
            img_tensors.append(
                to_tensor(img.convert("RGB" if channels == 3 else "L"))
            )

    X = torch.stack(img_tensors)

    if not has_labels:
        return X

    # кодируем названия классов цифрами от 0 до 9
    classes = sorted(df["label"].unique())
    classes_nums = {name: i for i, name in enumerate(classes)}

    # превращаем метки в тензор из номеров классов
    y = torch.tensor(df["label"].map(classes_nums).tolist())

    return X, y

def train_dataset_split(ds_train: TensorDataset, train_size: float, random_seed: int = 52) -> tuple[Subset, Subset]:
    n = len(ds_train)
    train_size = int(train_size * n)
    test_size = n - train_size

    g = torch.Generator().manual_seed(random_seed) # фиксируем рандом для random_split
    ds_train, ds_test = random_split(ds_train, [train_size, test_size], generator=g)
    return ds_train, ds_test

def create_dataloader(
    ds: Dataset, batch_size: int = 128, shuffle: bool = False,
    random_seed: int = 52, num_workers: int = 0,
    persistent_workers: bool = False,
) -> DataLoader:
    g = torch.Generator().manual_seed(random_seed) # фиксируем рандом для shuffle
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=g,
        num_workers=num_workers, persistent_workers=persistent_workers,
    )
    return dl


def load_training_data(data_dir, seed):
    x, y = images_to_tensor(data_dir / "train", data_dir / "trainLabels.csv")
    train, val = train_dataset_split(TensorDataset(x, y), train_size=0.8, random_seed=seed)
    mean, std = calculate_mean_std(train)
    return train, val, mean, std


def make_loaders(data, device, *, batch_size, seed, workers, augment):
    train, val, mean, std = data
    train_ready = AutoAugumentationDataset(train) if augment else train
    preprocess = BatchStandardizer(mean, std, device)
    train_loader = create_dataloader(train_ready, batch_size, shuffle=True, random_seed=seed, num_workers=workers, persistent_workers=workers > 0)
    val_loader = create_dataloader(val, batch_size, random_seed=seed)
    return train_loader, val_loader, preprocess
