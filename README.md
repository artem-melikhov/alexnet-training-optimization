
# AlexNet Training Optimization on CIFAR-10

Репозиторий экспериментов на NVIDIA RTX 2080 Ti. Исходная модель взята из
[проекта CIFAR-10](https://github.com/artem-melikhov/CIFAR-10/tree/85bce29cc2244516b6031185524d65dd693d33a7).
История этого репозитория реконструирует этапы экспериментов в воспроизводимом виде.

Для запуска нужны Python 3.10+, CUDA-версия PyTorch с подходящей torchvision и
данные в `data/train/` и `data/trainLabels.csv`. Выполните
`python -m pip install -r requirements.txt` в окружении ядра ноутбука.
Откройте `notebooks/02_training.ipynb` и выполните ячейки сверху вниз.
