import torch
from model_enhanced import EnhancedAttentionUNet

model = EnhancedAttentionUNet(n_classes=19)
params = sum(p.numel() for p in model.parameters())
print(f"Enhanced model parameters: {params:,}")
if params <= 1821085:
    print(f"✓ Under limit by {1821085 - params:,}")
else:
    print(f"✗ Over limit by {params - 1821085:,}")