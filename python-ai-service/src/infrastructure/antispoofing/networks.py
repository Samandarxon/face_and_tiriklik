"""MiniFASNet arxitekturasi (Silent-Face-Anti-Spoofing, MiniVision AI — MIT litsenziya).

Bu fayl faqat model grafini quradi. Vaznlar (`*.pth`) alohida yuklab olinadi:
`python-ai-service/scripts/download_models.sh` ga qarang.

Original manba: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
"""

from __future__ import annotations

import torch
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Linear,
    Module,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)

# Har bir MiniFASNet variantidagi kanal konfiguratsiyasi (1.8M parametrli modellar).
keep_dict = {
    "1.8M": [
        32, 32, 103, 103, 64, 13, 13, 64, 26, 26, 64, 13, 13, 64, 52, 52, 64,
        231, 231, 128, 154, 154, 128, 52, 52, 128, 26, 26, 128, 52, 52, 128,
        26, 26, 128, 26, 26, 128, 308, 308, 128, 26, 26, 128, 26, 26, 128,
        512, 512,
    ],
    "1.8M_": [
        32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64,
        231, 231, 128, 231, 231, 128, 52, 52, 128, 26, 26, 128, 77, 77, 128,
        26, 26, 128, 26, 26, 128, 308, 308, 128, 26, 26, 128, 26, 26, 128,
        512, 512,
    ],
}


class Flatten(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride,
                           padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride,
                           padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class SEModule(Module):
    """Squeeze-and-Excitation: kanal e'tibori (attention)."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(channels // reduction)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = BatchNorm2d(channels)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.sigmoid(self.bn2(self.fc2(x)))
        return module_input * x


class DepthWise(Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2),
                 padding=(1, 1), groups=1):
        super().__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = ConvBlock(c1_in, c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = ConvBlock(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = LinearBlock(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        short_cut = x if self.residual else None
        out = self.project(self.conv_dw(self.conv(x)))
        return short_cut + out if self.residual else out


class DepthWiseSE(Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2),
                 padding=(1, 1), groups=1, se_reduct=8):
        super().__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = ConvBlock(c1_in, c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = ConvBlock(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = LinearBlock(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
        if residual:
            self.se_module = SEModule(c3_out, se_reduct)

    def forward(self, x):
        short_cut = x if self.residual else None
        out = self.project(self.conv_dw(self.conv(x)))
        if self.residual:
            out = short_cut + self.se_module(out)
        return out


class Residual(Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        modules = [
            DepthWise(c1[i], c2[i], c3[i], residual=True, kernel=kernel,
                      padding=padding, stride=stride, groups=groups)
            for i in range(num_block)
        ]
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class ResidualSE(Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1),
                 padding=(1, 1), se_reduct=4):
        super().__init__()
        modules = []
        for i in range(num_block):
            # SE bloki faqat oxirgi qatlamga qo'yiladi (original implementatsiyaga mos).
            block_cls = DepthWiseSE if i == num_block - 1 else DepthWise
            kwargs = {"se_reduct": se_reduct} if i == num_block - 1 else {}
            modules.append(
                block_cls(c1[i], c2[i], c3[i], residual=True, kernel=kernel,
                          padding=padding, stride=stride, groups=groups, **kwargs)
            )
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MiniFASNet(Module):
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7), drop_p=0.0,
                 num_classes=3, img_channel=3):
        super().__init__()
        self.embedding_size = embedding_size

        self.conv1 = ConvBlock(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(keep[0], keep[1], kernel=(3, 3), stride=(1, 1),
                                  padding=(1, 1), groups=keep[1])

        self.conv_23 = DepthWise((keep[1], keep[2]), (keep[2], keep[3]), (keep[3], keep[4]),
                                 kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])

        c1, c2, c3 = self._stage_channels(keep, start=4, num_block=4)
        self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep[4],
                               kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_34 = DepthWise((keep[16], keep[17]), (keep[17], keep[18]), (keep[18], keep[19]),
                                 kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])

        c1, c2, c3 = self._stage_channels(keep, start=19, num_block=6)
        self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep[19],
                               kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_45 = DepthWise((keep[37], keep[38]), (keep[38], keep[39]), (keep[39], keep[40]),
                                 kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])

        c1, c2, c3 = self._stage_channels(keep, start=40, num_block=2)
        self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep[40],
                               kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_6_sep = ConvBlock(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = LinearBlock(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel,
                                     stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = torch.nn.Dropout(p=drop_p)
        self.prob = Linear(embedding_size, num_classes, bias=False)

    @staticmethod
    def _stage_channels(keep, start: int, num_block: int):
        """Residual bosqichi uchun (in, out) kanal juftliklarini hosil qiladi."""
        c1, c2, c3 = [], [], []
        for i in range(num_block):
            base = start + i * 3
            c1.append((keep[base], keep[base + 1]))
            c2.append((keep[base + 1], keep[base + 2]))
            c3.append((keep[base + 2], keep[base + 3]))
        return c1, c2, c3

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        return self.prob(out)


class MiniFASNetSE(MiniFASNet):
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7), drop_p=0.75,
                 num_classes=4, img_channel=3):
        super().__init__(keep=keep, embedding_size=embedding_size, conv6_kernel=conv6_kernel,
                         drop_p=drop_p, num_classes=num_classes, img_channel=img_channel)

        c1, c2, c3 = self._stage_channels(keep, start=4, num_block=4)
        self.conv_3 = ResidualSE(c1, c2, c3, num_block=4, groups=keep[4],
                                 kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1, c2, c3 = self._stage_channels(keep, start=19, num_block=6)
        self.conv_4 = ResidualSE(c1, c2, c3, num_block=6, groups=keep[19],
                                 kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1, c2, c3 = self._stage_channels(keep, start=40, num_block=2)
        self.conv_5 = ResidualSE(c1, c2, c3, num_block=2, groups=keep[40],
                                 kernel=(3, 3), stride=(1, 1), padding=(1, 1))


def MiniFASNetV1(embedding_size=128, conv6_kernel=(7, 7), drop_p=0.2, num_classes=3, img_channel=3):
    return MiniFASNet(keep_dict["1.8M"], embedding_size, conv6_kernel, drop_p, num_classes, img_channel)


def MiniFASNetV2(embedding_size=128, conv6_kernel=(7, 7), drop_p=0.2, num_classes=3, img_channel=3):
    return MiniFASNet(keep_dict["1.8M_"], embedding_size, conv6_kernel, drop_p, num_classes, img_channel)


def MiniFASNetV1SE(embedding_size=128, conv6_kernel=(7, 7), drop_p=0.75, num_classes=3, img_channel=3):
    return MiniFASNetSE(keep_dict["1.8M"], embedding_size, conv6_kernel, drop_p, num_classes, img_channel)


def MiniFASNetV2SE(embedding_size=128, conv6_kernel=(7, 7), drop_p=0.75, num_classes=3, img_channel=3):
    return MiniFASNetSE(keep_dict["1.8M_"], embedding_size, conv6_kernel, drop_p, num_classes, img_channel)


MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}
