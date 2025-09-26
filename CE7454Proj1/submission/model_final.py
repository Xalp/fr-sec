import torch
import torch.nn as nn
from model_utils import *


class FinalAttentionUNet(nn.Module):
    def __init__(
        self,
        n_classes=19,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(FinalAttentionUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        # Custom filters to target ~1.8M parameters
        # Between feature_scale=4 (1.96M) and feature_scale=5 (1.25M)
        # We need something around 4.3 scale
        base_filters = [64, 128, 256, 512, 1024]
        feature_scale = 4.3
        filters = [int(x / feature_scale) for x in base_filters]
        # This gives approximately: [14, 29, 59, 118, 237]
        # Let's use cleaner numbers
        filters = [15, 30, 60, 119, 238]

        # Encoder
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # Decoder with attention
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # Final conv
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        # Encoder
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        
        # Decoder
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final


# Let's also create one with exact parameter tuning
class TunedAttentionUNet(nn.Module):
    def __init__(
        self,
        n_classes=19,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(TunedAttentionUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        # Fine-tuned to get close to 1.8M
        # Starting from feature_scale=5 (1.25M) and increasing channels
        filters = [14, 28, 56, 112, 224]  # Slightly larger than scale=5
        
        # Encoder
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # Decoder with attention
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # Add extra convolutions to use more parameters
        self.extra_conv3 = nn.Sequential(
            nn.Conv2d(filters[2], filters[2], 3, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        )
        self.extra_conv2 = nn.Sequential(
            nn.Conv2d(filters[1], filters[1], 3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )
        self.extra_conv1 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], 3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )

        # Final conv
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        # Encoder
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        
        # Decoder with extra convolutions
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up3 = self.extra_conv3(up3)
        
        up2 = self.up_concat2(conv2, up3)
        up2 = self.extra_conv2(up2)
        
        up1 = self.up_concat1(conv1, up2)
        up1 = self.extra_conv1(up1)

        final = self.final(up1)

        return final


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test both models
    print("Testing FinalAttentionUNet:")
    model1 = FinalAttentionUNet(n_classes=19)
    params1 = count_parameters(model1)
    print(f"Parameters: {params1:,}")
    if params1 <= 1821085:
        print(f"✓ Under limit by {1821085 - params1:,}")
    else:
        print(f"✗ Over limit by {params1 - 1821085:,}")
    
    print("\nTesting TunedAttentionUNet:")
    model2 = TunedAttentionUNet(n_classes=19)
    params2 = count_parameters(model2)
    print(f"Parameters: {params2:,}")
    if params2 <= 1821085:
        print(f"✓ Under limit by {1821085 - params2:,}")
    else:
        print(f"✗ Over limit by {params2 - 1821085:,}")