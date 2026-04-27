import torch.nn as nn


class BoWClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)