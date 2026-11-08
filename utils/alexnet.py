import torch
from torch import nn


class AlexNet(nn.Module):
    """Упрощённая адаптация AlexNet для изображений CIFAR-10 размером 32x32."""

    def __init__(
        self,
        in_channels: int = 3,
        output_size: int = 10,
        image_size: tuple[int, int] = (32, 32)
    ):

        super().__init__()

        modules = []
        conv_layers = [96, 256, 384, 384, 256]

        in_ = in_channels # количество входных каналов для свёртки
        for i in range(len(conv_layers)): # цикл добавления свёрточных блоков (смотреть схему)

            # свёртка
            if i == 1:
                modules.append(nn.Conv2d(
                    in_channels=in_,
                    out_channels=conv_layers[i],
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    ))
            else:
                modules.append(nn.Conv2d(
                    in_channels=in_,
                    out_channels=conv_layers[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    ))
            in_ = conv_layers[i]

            modules.append(nn.ReLU()) # слой активации после какждой свёртки

            if i in [0, 1]:
                modules.append(nn.LocalResponseNorm(size=5)) # значение size=5 взято из оригинальной версии

            if i in [0, 1, 4]: # в блоках 1, 2, 5 есть maxpool
                modules.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        #вычиляем на искусственном примере размер картинки, который получился после всех свёрток
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, *image_size)
            in_features = nn.Sequential(*modules)(sample).flatten(1).shape[1]

        # после всех свёрточных слоёв распрямляем значения в вектор, чтобы подать на вход линейному слою
        modules.append(nn.Flatten())
        # после выпрямления получаем вектор из 4096 чисел

        # 1 линейный слой на 512 нейронов
        modules.append(nn.Linear(in_features, 512))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.5))

        # 2 линейный слой на 256 нейронов
        modules.append(nn.Linear(512, 256))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.5))

        # в конце добавляем выходной слой, дающий предсказания (логиты) для каждого из классов
        modules.append(nn.Linear(256, output_size))

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)
