from typing import Dict

import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: Dict, path: str) -> None:
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict:
    return torch.load(path, map_location=map_location)


def freeze_bn(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

