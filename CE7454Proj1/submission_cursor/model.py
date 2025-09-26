from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, stride=1)

        self.residual = None
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.residual is not None:
            identity = self.residual(x)
        return nn.functional.relu(out + identity)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: List[int]) -> None:
        super().__init__()
        modules: List[nn.Module] = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        for rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(self.convs), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: List[torch.Tensor] = []
        for idx, conv in enumerate(self.convs):
            if idx == len(self.convs) - 1:
                pooled = conv(x)
                pooled = nn.functional.interpolate(pooled, size=x.shape[2:], mode="bilinear", align_corners=False)
                res.append(pooled)
            else:
                res.append(conv(x))
        x = torch.cat(res, dim=1)
        return self.project(x)


class LightweightSegmentationNet(nn.Module):
    def __init__(self, num_classes: int = 16) -> None:
        super().__init__()
        encoder_channels = [24, 48, 128, 256]
        self.stem = nn.Sequential(
            conv_bn_relu(3, encoder_channels[0], kernel_size=3, stride=2),
            conv_bn_relu(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1),
        )

        self.block1 = nn.Sequential(
            ResidualBlock(encoder_channels[0], encoder_channels[0], stride=1),
            ResidualBlock(encoder_channels[0], encoder_channels[0], stride=1),
        )
        self.block2 = nn.Sequential(
            ResidualBlock(encoder_channels[0], encoder_channels[1], stride=2),
            ResidualBlock(encoder_channels[1], encoder_channels[1], stride=1),
        )
        self.block3 = nn.Sequential(
            ResidualBlock(encoder_channels[1], encoder_channels[2], stride=2),
            ResidualBlock(encoder_channels[2], encoder_channels[2], stride=1),
        )
        self.block4 = nn.Sequential(
            ResidualBlock(encoder_channels[2], encoder_channels[3], stride=2),
            ResidualBlock(encoder_channels[3], encoder_channels[3], stride=1),
        )

        self.aspp = ASPP(encoder_channels[3], 128, atrous_rates=[1, 6, 12])

        self.decoder_conv1 = conv_bn_relu(encoder_channels[2] + 128, 96, kernel_size=3)
        self.decoder_conv2 = conv_bn_relu(encoder_channels[1] + 96, 64, kernel_size=3)
        self.decoder_conv3 = conv_bn_relu(encoder_channels[0] + 64, 48, kernel_size=3)
        self.classifier = nn.Conv2d(48, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x = self.stem(x)
        feat1 = self.block1(x)
        feat2 = self.block2(feat1)
        feat3 = self.block3(feat2)
        feat4 = self.block4(feat3)

        x = self.aspp(feat4)

        x = nn.functional.interpolate(x, size=feat3.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, feat3], dim=1)
        x = self.decoder_conv1(x)

        x = nn.functional.interpolate(x, size=feat2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, feat2], dim=1)
        x = self.decoder_conv2(x)

        x = nn.functional.interpolate(x, size=feat1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, feat1], dim=1)
        x = self.decoder_conv3(x)

        x = nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.classifier(x)


def _get_gn_groups(num_channels: int) -> int:
    for group in (32, 16, 8, 4, 2):
        if num_channels % group == 0:
            return group
    return 1


def conv_gn_gelu(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.GroupNorm(_get_gn_groups(out_channels), out_channels),
        nn.GELU(),
    )


class ResidualDSBlockGN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.dw_norm = nn.GroupNorm(_get_gn_groups(in_channels), in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_norm = nn.GroupNorm(_get_gn_groups(out_channels), out_channels)
        self.activation = nn.GELU()

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_get_gn_groups(out_channels), out_channels),
            )
        else:
            self.residual = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.depthwise(x)
        out = self.dw_norm(out)
        out = self.activation(out)
        out = self.pointwise(out)
        out = self.pw_norm(out)
        if self.residual is not None:
            identity = self.residual(x)
        out = out + identity
        return self.activation(out)


class LightweightSegmentationNetV2(nn.Module):
    def __init__(self, num_classes: int = 16) -> None:
        super().__init__()
        c1, c2, c3 = 48, 96, 128

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_get_gn_groups(c1), c1),
            nn.GELU(),
            ResidualDSBlockGN(c1, c1),
        )

        self.enc1 = ResidualDSBlockGN(c1, c1)

        self.enc2_down = ResidualDSBlockGN(c1, c2, stride=2)
        self.enc2 = ResidualDSBlockGN(c2, c2)

        self.enc3_down = ResidualDSBlockGN(c2, c3, stride=2)
        self.enc3 = ResidualDSBlockGN(c3, c3)
        self.context = ResidualDSBlockGN(c3, c3, dilation=2)

        self.dec1_reduce = conv_gn_gelu(c3 + c2, c2)
        self.dec1_block = ResidualDSBlockGN(c2, c2)

        self.dec2_reduce = conv_gn_gelu(c2 + c1, 64)
        self.dec2_block = ResidualDSBlockGN(64, 64)

        self.dec3_reduce = conv_gn_gelu(64, c1)
        self.dec3_block = ResidualDSBlockGN(c1, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]

        x = self.stem(x)
        skip1 = self.enc1(x)

        x = self.enc2_down(skip1)
        skip2 = self.enc2(x)

        x = self.enc3_down(skip2)
        x = self.enc3(x)
        x = self.context(x)

        x = F.interpolate(x, size=skip2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec1_reduce(x)
        x = self.dec1_block(x)

        x = F.interpolate(x, size=skip1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec2_reduce(x)
        x = self.dec2_block(x)

        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = self.dec3_reduce(x)
        x = self.dec3_block(x)

        return self.head(x)


def build_model(config: Dict) -> nn.Module:
    num_classes = config.get("num_classes", 16)
    model_type = config.get("model_type", "v1")

    if model_type == "v2":
        return LightweightSegmentationNetV2(num_classes=num_classes)

    return LightweightSegmentationNet(num_classes=num_classes)

