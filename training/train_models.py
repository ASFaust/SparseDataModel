import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle

class MLPRegressor(nn.Module):
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

def train_model(name, dataset, input_dim, epochs=1000, batch_size=128, device="cuda"):
    X, y = zip(*dataset)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    # Add second-degree polynomial features (x_i * x_j for all i <= j)
    #n_samples, n_features = X.shape
    #poly_features = [X]  # original features
    #for i in range(n_features):
    #    for j in range(i, n_features):
    #        poly_features.append((X[:, i] * X[:, j]).unsqueeze(1))
    #X = torch.cat(poly_features, dim=1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

    full_ds = TensorDataset(X, y)
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLPRegressor(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ma_loss = 0.0
    ma_abs_diff = 0.0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = (out - yb).pow(2).mean()
            mean_abs_diff = (out - yb).abs().mean()
            ma_abs_diff = 0.999 * ma_abs_diff + 0.001 * mean_abs_diff.item() if epoch > 0 else mean_abs_diff.item()
            loss.backward()
            optimizer.step()
            ma_loss = 0.999 * ma_loss + 0.001 * loss.item() if epoch > 0 else loss.item()
            #print(f"\r{name} Epoch {epoch + 1}/{epochs}, Train Loss: {ma_loss:.4f}, Mean Abs Diff: {ma_abs_diff:.4f}", end="")

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append((model(xb) - yb).pow(2).mean().item())
            val_loss = sum(val_losses) / len(val_losses)
        print(f"\r{name} Epoch {epoch + 1}/{epochs}, Train Loss: {ma_loss:.4f}, Val Loss: {val_loss:.4f}, Mean Abs Diff: {ma_abs_diff:.4f}",end="", flush=True)


    print(f"\n{name} training complete.")
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)

    trained_models = {}
    for key in data:
        #if key == "corr_bb":
        #    continue
        dim = len(data[key][0][0])
        # calculate polynomial feature dimension: original + (n_features * (n_features + 1)) / 2
        #dim = int(dim + dim * (dim + 1) / 2)
        trained_models[key] = train_model(key, data[key], input_dim=dim, device=device)

    torch.save(trained_models, 'trained_models.pt')
