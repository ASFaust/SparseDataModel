import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle



# Define a small MLP regressor
# All estimators are now for the correlation matrix, so we can use tanh activation for the output
# since correlation values are in the range [-1, 1].
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden)
        self.att1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.att2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        h1 = self.sigmoid(self.att1(x)) * self.l1(x)
        h2 = self.att2(self.sigmoid(h1)) * self.l2(h1)
        return self.tanh(self.out(h2))

# Training function
def train_model(name, dataset, input_dim, epochs=400, batch_size=128):
    X, y = zip(*dataset)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = MLPRegressor(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ma_loss = 0.0
    ma_abs_diff = 0.0
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = (out - yb).pow(2).mean()
            mean_abs_diff = (out - yb).abs().mean()
            ma_abs_diff = 0.99 * ma_abs_diff + 0.01 * mean_abs_diff.item() if epoch > 0 else mean_abs_diff.item()
            #print(f"\r{name} Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}", end="")
            loss.backward()
            optimizer.step()
            ma_loss = 0.99 * ma_loss + 0.01 * loss.item() if epoch > 0 else loss.item()
        print(f"\r{name} Epoch {epoch+1}/{epochs}, Loss: {ma_loss:.4f}, mean abs diff: {ma_abs_diff:.4f}", end="")

    print(f"\n{name} training complete.")
    return model

if __name__ == "__main__":
    # Load training data
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Train all 6 estimators
    trained_models = {}
    for key in data:
        dim = len(data[key][0][0])
        trained_models[key] = train_model(key, data[key], input_dim=dim)

    # Optional: Save models
    torch.save(trained_models, 'trained_models.pt')
