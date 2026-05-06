import torch
import torch.nn as nn

class PricePredictor(nn.Module):
    def __init__(self, input_size=20):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        return self.model(x)
