import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle

# Load training data
with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Define a small MLP regressor
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden)
        self.att1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.att2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.sigmoid(self.att1(x)) * self.l1(x)
        h2 = self.att2(self.sigmoid(h1)) * self.l2(h1)
        return self.out(h2)

# Training function
def train_model(name, dataset, input_dim, epochs=100, batch_size=128):
    X, y = zip(*dataset)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = MLPRegressor(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            print(f"\r{name} Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}", end="")
            loss.backward()
            optimizer.step()

    print(f"{name}: Final training loss = {loss.item():.4f}")
    return model

# Train all 6 estimators
trained_models = {}
for key in data:
    dim = len(data[key][0][0])
    trained_models[key] = train_model(key, data[key], input_dim=dim)

# Optional: Save models
torch.save(trained_models, 'trained_models.pt')
