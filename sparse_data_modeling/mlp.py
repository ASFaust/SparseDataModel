import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(input_dim, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.l4 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(x)
        h3 = h1 / (self.sigmoid(h2) + 1e-7)
        h4 = self.l3(h3)
        h5 = self.l4(h4)
        h6 = h5 / (self.sigmoid(h4) + 1e-7)
        return self.tanh(self.out(h6))