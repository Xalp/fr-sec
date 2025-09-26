from dataclasses import dataclass


@dataclass
class TrainingConfig:
    num_classes: int = 16
    batch_size: int = 8
    num_epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    save_every: int = 5
    val_split: float = 0.1

