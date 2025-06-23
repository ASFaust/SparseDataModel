import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle

torch.set_float32_matmul_precision('high')

class MLP(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(input_dim, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.l4 = nn.Linear(hidden, hidden)

        self.out = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.c1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)
        self.c2 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        h1 = self.l1(x) / (self.elu(self.l2(x)) + 1.000001 + torch.abs(self.c1))
        h2 = self.l3(h1) / (self.elu(self.l4(h1)) + 1.000001 + torch.abs(self.c2))
        return self.tanh(self.out(h2))

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

    #small_ds, _ = random_split(full_ds, [10000, n_total - 10000])  # Use a small subset for quick training
    small_ds = full_ds  # Use the full dataset for training
    n_train = int(0.8 * len(small_ds))
    n_val = len(small_ds) - n_train
    train_ds, val_ds = random_split(small_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLP(input_dim).to(device)
    #model.compile(mode="max-autotune-no-cudagraphs", dynamic=False, fullgraph=True)
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
            max_abs_diff = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append((model(xb) - yb).pow(2).mean().item())
                max_abs_diff = max(max_abs_diff, (model(xb) - yb).abs().mean().item())
            val_loss = sum(val_losses) / len(val_losses)
        print(f"\r{name} Epoch {epoch + 1}/{epochs}, Train Loss: {ma_loss:.4f}, Val Loss: {val_loss:.4f}, Mean Abs Diff: {ma_abs_diff:.4f}, Max Abs Diff: {max_abs_diff:.4f}", end="", flush=True)


    print(f"\n{name} training complete.")
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)

    n_epochs = {
        "corr_bb": 1000,
        "corr_gg": 250,
        "corr_gb": 250,
    }

    trained_models = {}
    for key in data:
        #if key == "corr_bb":
        #    continue
        dim = len(data[key][0][0])
        #print number of datapoints in the dataset

        # calculate polynomial feature dimension: original + (n_features * (n_features + 1)) / 2
        #dim = int(dim + dim * (dim + 1) / 2)
        trained_models[key] = train_model(key, data[key], input_dim=dim, device=device, epochs=n_epochs[key])
        #print c1 and c2 parameters
        print(f"{key} model trained with input dimension {dim}. c1: {trained_models[key].c1.item()}, c2: {trained_models[key].c2.item()}")

    torch.save(trained_models, 'trained_models.pt')
