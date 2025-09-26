import torch
from model import AttentionUNet
from model_optimized import OptimizedAttentionUNet, PreciseAttentionUNet

print("Testing different model configurations:\n")

# Test original with different feature scales
print("Original AttentionUNet with different feature_scales:")
for scale in [4, 4.5, 5]:
    if scale == 4.5:
        # Approximate between 4 and 5
        filters = [57, 114, 228, 456, 912]
        model = AttentionUNet(feature_scale=4, n_classes=19)
        # This is just to show the concept
        continue
    else:
        model = AttentionUNet(feature_scale=scale, n_classes=19)
    params = sum(p.numel() for p in model.parameters())
    print(f"  feature_scale={scale}: {params:,} parameters")

print("\nOptimizedAttentionUNet:")
model = OptimizedAttentionUNet(n_classes=19)
params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {params:,}")
if params <= 1821085:
    print(f"  ✓ Under limit by {1821085 - params:,}")
else:
    print(f"  ✗ Over limit by {params - 1821085:,}")

print("\nPreciseAttentionUNet:")
model = PreciseAttentionUNet(n_classes=19)
params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {params:,}")
if params <= 1821085:
    print(f"  ✓ Under limit by {1821085 - params:,}")
else:
    print(f"  ✗ Over limit by {params - 1821085:,}")