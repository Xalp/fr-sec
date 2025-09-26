from model import AttentionUNet

model = AttentionUNet(feature_scale=4.5, n_classes=19)
params = sum(p.numel() for p in model.parameters())
print(f"Model with feature_scale=4.5: {params:,} parameters")
print(f"Under limit by: {1821085 - params:,} parameters")
print(f"Using {(params/1821085)*100:.1f}% of allowed parameters")