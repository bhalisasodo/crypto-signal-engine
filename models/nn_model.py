import torch
import torch.nn as nn

class PricePredictor(nn.Module):
    def __init__(self, input_size=4, seq_len=20, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, self.seq_len, self.input_size)

        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])

    def save(self, path):
        torch.save({
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_size=checkpoint["input_size"],
            seq_len=checkpoint["seq_len"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
