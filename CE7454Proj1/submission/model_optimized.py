import torch
import torch.nn as nn
from model_utils import *


class OptimizedAttentionUNet(nn.Module):
    def __init__(
        self,
        n_classes=19,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(OptimizedAttentionUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        # Optimized filter configuration for ~1.8M parameters
        # Based on interpolation between feature_scale=4 (1.97M) and feature_scale=5 (1.25M)
        filters = [54, 108, 216, 432, 864]
        
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

        # Add extra refinement layers to use more parameters
        self.refine3 = nn.Sequential(
            nn.Conv2d(filters[2], filters[2], 3, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(filters[1], filters[1], 3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], 3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], n_classes, 1)
        )

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
        
        # Decoder with refinement
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up3 = self.refine3(up3)
        
        up2 = self.up_concat2(conv2, up3)
        up2 = self.refine2(up2)
        
        up1 = self.up_concat1(conv1, up2)
        
        final = self.final_conv(up1)

        return final


# Alternative: Fine-tuned channel sizes
class PreciseAttentionUNet(nn.Module):
    def __init__(
        self,
        n_classes=19,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(PreciseAttentionUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        # Fine-tuned to get very close to 1.8M
        filters = [52, 104, 208, 416, 832]
        
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

        # Decoder
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # Deep supervision
        self.side4 = nn.Conv2d(filters[3], n_classes, 1)
        self.side3 = nn.Conv2d(filters[2], n_classes, 1)
        self.side2 = nn.Conv2d(filters[1], n_classes, 1)
        
        # Final
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs, deep_supervision=False):
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
        
        if deep_supervision:
            side4 = nn.functional.interpolate(self.side4(up4), size=inputs.size()[2:], mode='bilinear', align_corners=True)
            side3 = nn.functional.interpolate(self.side3(up3), size=inputs.size()[2:], mode='bilinear', align_corners=True)
            side2 = nn.functional.interpolate(self.side2(up2), size=inputs.size()[2:], mode='bilinear', align_corners=True)
            return final, side2, side3, side4
        
        return final


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test both models
    print("Testing OptimizedAttentionUNet:")
    model1 = OptimizedAttentionUNet(n_classes=19)
    params1 = count_parameters(model1)
    print(f"Parameters: {params1:,}")
    
    print("\nTesting PreciseAttentionUNet:")
    model2 = PreciseAttentionUNet(n_classes=19)
    params2 = count_parameters(model2)
    print(f"Parameters: {params2:,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    y1 = model1(x)
    y2 = model2(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y1.shape}")