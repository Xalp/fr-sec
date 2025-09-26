import torch
from model import AttentionUNet

for scale in [3, 4, 5, 6, 7, 8]:
    model = AttentionUNet(feature_scale=scale, n_classes=19)
    params = sum(p.numel() for p in model.parameters())
    print(f"feature_scale={scale}: {params:,} parameters")
    if params <= 1821085:
        print(f"  ✓ Under limit by {1821085 - params:,}")
    else:
        print(f"  ✗ Over limit by {params - 1821085:,}")