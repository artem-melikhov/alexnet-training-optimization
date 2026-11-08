
import torch
from PIL import Image
from torchvision.transforms import ToTensor, v2
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

        # AutoAugment ожидает тензор uint8 со значениями от 0 до 255.
        image = (image * 255).round().to(torch.uint8)
        image = self.augment(image)

        # Возвращаем формат float32 со значениями от 0 до 1, как после ToTensor().
        image = image.to(torch.float32) / 255
        return image, label


def calculate_mean_std(ds_train: Subset) -> tuple[torch.Tensor, torch.Tensor]:
    """Считает среднее и стандартное отклонение по обучающим картинкам."""

    # берём только train, чтобы картинки из val не влияли на статистики
    X_train = ds_train.dataset.tensors[0][ds_train.indices]

    mean = X_train.mean(dim=(0, 2, 3))
    std = X_train.std(dim=(0, 2, 3), correction=0)

    return mean, std


class StandardizationDataset(Dataset):
    """Стандартизует картинки по mean и std обучающей выборки."""

    def __init__(self, dataset: Dataset, mean: torch.Tensor, std: torch.Tensor):
        self.dataset = dataset
        # добавляем размеры высоты и ширины, чтобы значения применились к каждому каналу
        self.mean = mean.reshape(-1, 1, 1)
        self.std = std.reshape(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        data = self.dataset[index]

        if isinstance(data, tuple):
            image, label = data
            return (image - self.mean) / self.std, label

        return (data - self.mean) / self.std


def images_to_tensor(
    images_path: str,
    csv_path: str,
    channels: int = 3,
    has_labels: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

    # читаем метки правильных классов для фото они сопоставлены с ними по image_id, 
    # который указан в названии файла картинки(для случая когда эти метки есть, то есть train), 
    # иначе читаем id которые идут в файле sample_submission, 
    # чтобы дать предсказания для картинок в том же порядке
    df = pd.read_csv(csv_path)
    to_tensor = ToTensor()
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

def create_dataloader(ds: Dataset, batch_size: int = 128, shuffle: bool = False, random_seed: int = 52) -> DataLoader:
    g = torch.Generator().manual_seed(random_seed) # фиксируем рандом для shuffle
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)
    return dl


def load_training_data(data_dir, seed):
    x, y = images_to_tensor(data_dir / "train", data_dir / "trainLabels.csv")
    train, val = train_dataset_split(TensorDataset(x, y), train_size=0.8, random_seed=seed)
    mean, std = calculate_mean_std(train)
    return train, val, mean, std


def make_loaders(data, device, *, batch_size, seed, workers, augment):
    train, val, mean, std = data
    train_ready = AutoAugumentationDataset(train) if augment else train
    train_ready = StandardizationDataset(train_ready, mean, std)
    val = StandardizationDataset(val, mean, std)
    preprocess = None
    train_loader = create_dataloader(train_ready, batch_size, shuffle=True, random_seed=seed)
    val_loader = create_dataloader(val, batch_size, random_seed=seed)
    return train_loader, val_loader, preprocess
