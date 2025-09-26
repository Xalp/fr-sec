import torch
import torch.nn as nn
from model_utils import *


class EnhancedAttentionUNet(nn.Module):
    def __init__(
        self,
        n_classes=19,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(EnhancedAttentionUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        # Custom filter configuration to maximize parameter usage
        # Target: ~1.8M parameters
        filters = [48, 96, 192, 384, 768]
        
        # Initial conv with residual
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.res1 = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.res2 = nn.Conv2d(filters[1], filters[1], 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.res3 = nn.Conv2d(filters[2], filters[2], 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Enhanced center with extra convolutions
        self.center = nn.Sequential(
            unetConv2(filters[3], filters[4], self.is_batchnorm),
            nn.Conv2d(filters[4], filters[4], 3, padding=1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True),
        )

        # Upsampling with attention
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # Multi-scale feature fusion
        self.side3 = nn.Conv2d(filters[2], n_classes, 1)
        self.side2 = nn.Conv2d(filters[1], n_classes, 1)
        self.side1 = nn.Conv2d(filters[0], n_classes, 1)
        
        # Final fusion
        self.final_conv = nn.Conv2d(filters[0] + n_classes * 3, filters[0], 3, padding=1)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        # Encoder with residual connections
        conv1 = self.conv1(inputs)
        conv1 = conv1 + self.res1(conv1)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        conv2 = conv2 + self.res2(conv2)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        conv3 = conv3 + self.res3(conv3)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        
        # Decoder
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Multi-scale predictions
        side3 = nn.functional.interpolate(self.side3(up3), size=inputs.size()[2:], mode='bilinear', align_corners=True)
        side2 = nn.functional.interpolate(self.side2(up2), size=inputs.size()[2:], mode='bilinear', align_corners=True)
        side1 = self.side1(up1)
        
        # Concatenate multi-scale features
        fusion = torch.cat([up1, side3, side2, side1], dim=1)
        fusion = self.final_conv(fusion)
        
        final = self.final(fusion)

        return final


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model and count parameters
    model = EnhancedAttentionUNet(n_classes=19)
    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")