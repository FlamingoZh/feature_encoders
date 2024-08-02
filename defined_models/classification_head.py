import torch
import torch.nn as nn

class MLP_head(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out = self.model(x)
        return out
