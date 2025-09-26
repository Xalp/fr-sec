import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_group_gn(channels, max_groups=8):
    """Get appropriate number of groups for GroupNorm"""
    for g in [8, 6, 4, 3, 2, 1]:
        if channels % g == 0 and g <= max_groups:
            return g
    return 1


class DSConvBlock(nn.Module):
    """Depthwise Separable Convolution Block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dw_kernel=3):
        super().__init__()
        self.use_residual = (in_channels == out_channels and stride == 1)
        
        # Pointwise -> GroupNorm -> GELU -> Depthwise -> GroupNorm -> GELU -> Pointwise -> GroupNorm
        self.pw1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.gn1 = nn.GroupNorm(get_group_gn(out_channels), out_channels)
        
        self.dw = nn.Conv2d(out_channels, out_channels, dw_kernel, stride, dw_kernel//2, groups=out_channels, bias=False)
        self.gn2 = nn.GroupNorm(get_group_gn(out_channels), out_channels)
        
        self.pw2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.gn3 = nn.GroupNorm(get_group_gn(out_channels), out_channels)
        
        self.act = nn.GELU()
        
        # Skip connection for first block if channels differ
        if in_channels != out_channels and stride == 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.use_residual = True
        else:
            self.skip = None
    
    def forward(self, x):
        identity = x
        
        out = self.pw1(x)
        out = self.gn1(out)
        out = self.act(out)
        
        out = self.dw(out)
        out = self.gn2(out)
        out = self.act(out)
        
        out = self.pw2(out)
        out = self.gn3(out)
        
        if self.use_residual:
            if self.skip is not None:
                identity = self.skip(identity)
            out = out + identity
        
        out = self.act(out)
        return out


class Downsample(nn.Module):
    """Downsample module between branches"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.gn = nn.GroupNorm(get_group_gn(out_channels), out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.gn(x)
        x = self.act(x)
        return x


class FusionModule(nn.Module):
    """HRNet-style fusion between high and low resolution branches"""
    def __init__(self, high_channels, low_channels):
        super().__init__()
        # High to low (downsample)
        self.high_to_low = nn.Sequential(
            nn.Conv2d(high_channels, high_channels, 3, 2, 1, groups=high_channels, bias=False),
            nn.Conv2d(high_channels, low_channels, 1, bias=False)
        )
        
        # Low to high (upsample)
        self.low_to_high = nn.Conv2d(low_channels, high_channels, 1, bias=False)
    
    def forward(self, high, low):
        # Downsample high to low resolution
        down = self.high_to_low(high)
        # Upsample low to high resolution
        up = F.interpolate(low, scale_factor=2, mode='bilinear', align_corners=True)
        up = self.low_to_high(up)
        
        # Fuse
        fused_high = high + up
        fused_low = low + down
        
        return fused_high, fused_low


class HRNetLiteFaceParser(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        
        # Reduce channel sizes to avoid OOM
        C_high = 160    # High resolution branch  
        C_low = 240     # Low resolution branch
        
        # Stem (full resolution)
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_high, 1, bias=False),
            nn.GroupNorm(get_group_gn(C_high), C_high),
            nn.GELU(),
            DSConvBlock(C_high, C_high, dw_kernel=5)  # Use 5x5 for sharper edges
        )
        
        # Stage 1 (1/1 resolution)
        self.stage1 = nn.Sequential(
            DSConvBlock(C_high, C_high),
            DSConvBlock(C_high, C_high),
            DSConvBlock(C_high, C_high)
        )
        
        # Stage 2 - create low resolution branch
        self.downsample = Downsample(C_high, C_low)
        
        # Stage 2 blocks (two branches)
        self.high_blocks1 = nn.Sequential(DSConvBlock(C_high, C_high), DSConvBlock(C_high, C_high))
        self.low_blocks1 = nn.Sequential(DSConvBlock(C_low, C_low), DSConvBlock(C_low, C_low))
        self.fusion1 = FusionModule(C_high, C_low)
        
        self.high_blocks2 = nn.Sequential(DSConvBlock(C_high, C_high), DSConvBlock(C_high, C_high))
        self.low_blocks2 = nn.Sequential(DSConvBlock(C_low, C_low), DSConvBlock(C_low, C_low))
        self.fusion2 = FusionModule(C_high, C_low)
        
        # Head
        C_fused = C_high * 2  # 320
        C_head = 256
        self.low_to_high_final = nn.Conv2d(C_low, C_high, 1, bias=False)
        self.head = nn.Sequential(
            DSConvBlock(C_fused, C_head),
            nn.Conv2d(C_head, C_head, 1, bias=False),
            nn.GroupNorm(get_group_gn(C_head), C_head),
            nn.GELU()
        )
        
        # Classifier
        self.classifier = nn.Conv2d(C_head, num_classes, 1)
        
        # Optional auxiliary heads
        self.edge_head = nn.Conv2d(C_high, 1, 1)  # Edge detection from Stage1
        self.aux_head = nn.Conv2d(C_low, num_classes, 1)  # Auxiliary segmentation from low branch
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Initialize classifier bias
        nn.init.constant_(self.classifier.bias, -np.log(self.classifier.out_channels - 1))
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Stage 1
        stage1_out = self.stage1(x)
        
        # Create low resolution branch
        high = stage1_out
        low = self.downsample(stage1_out)
        
        # Stage 2 - Block 1
        high = self.high_blocks1(high)
        low = self.low_blocks1(low)
        high, low = self.fusion1(high, low)
        
        # Stage 2 - Block 2
        high = self.high_blocks2(high)
        low = self.low_blocks2(low)
        high, low = self.fusion2(high, low)
        
        # Head - concatenate aligned features
        low_up = F.interpolate(low, scale_factor=2, mode='bilinear', align_corners=True)
        low_up = self.low_to_high_final(low_up)
        fused = torch.cat([high, low_up], dim=1)  # 24 + 24 = 48
        
        features = self.head(fused)
        out = self.classifier(features)
        
        # Auxiliary outputs (raw logits, no sigmoid)
        edge = self.edge_head(stage1_out)
        aux = self.aux_head(low)
        aux = F.interpolate(aux, scale_factor=2, mode='bilinear', align_corners=True)
        
        return out, edge, aux


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = HRNetLiteFaceParser(num_classes=19)
    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    out, edge, aux = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Edge shape: {edge.shape}")
    print(f"Aux shape: {aux.shape}")