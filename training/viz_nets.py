import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from train_models import MLP

torch.serialization.add_safe_globals({"MLP": MLP})
# Load trained models
trained_models = torch.load("trained_models.pt", weights_only=False)
for model in trained_models.values():
    model.cpu()
    model.eval()

# Load training data
with open("training_data.pkl", "rb") as f:
    training_data = pickle.load(f)

# Extract feature vectors from any of the datasets (they all share input format)


# Sweep configuration
n_samples = 5
n_points = 100

os.makedirs("sweeps", exist_ok=True)

colors = plt.cm.viridis(np.linspace(0, 1, n_samples))

for model_name, model in trained_models.items():
    # Get sample pool for this model
    sample_pool = np.array([x for x, _ in training_data[model_name]])
    input_dim = sample_pool.shape[1]
    targets = np.array([y for _, y in training_data[model_name]])

    for dim in range(input_dim):
        plt.figure()
        for sample_idx in range(n_samples):
            rand_idx = np.random.randint(len(sample_pool))
            base_input = sample_pool[rand_idx]
            true_y = training_data[model_name][rand_idx][1]
            sweep_min = sample_pool[:, dim].min()
            sweep_max = sample_pool[:, dim].max()
            sweep_vals = np.linspace(sweep_min, sweep_max, n_points)

            inputs = np.tile(base_input, (n_points, 1))
            inputs[:, dim] = sweep_vals

            with torch.no_grad():
                inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
                outputs = model(inputs_tensor).squeeze().numpy()

            plt.plot(sweep_vals, outputs, label=f'sample {sample_idx + 1}', color=colors[sample_idx])
            #plt.plot(base_input[dim], true_y, 'o', color=colors[sample_idx])
            #additionally, get all points in sample_pool that have the same value in all dimensions except dim
            all_dims_except_dim = np.arange(input_dim) != dim
            diff = sample_pool[:, all_dims_except_dim] - base_input[all_dims_except_dim]
            #print(f"Sample {sample_idx + 1}, dim {dim}, diff shape: {diff.shape}, base_input shape: {base_input[all_dims_except_dim].shape}")
            mask = np.all(np.abs(diff) < 0.0001, axis=1)
            plt.scatter(sample_pool[mask, dim], targets[mask], color=colors[sample_idx], alpha=0.2, s=20,
                        label=f'sample {sample_idx + 1} points')
            plt.title(f'{model_name} - Sweep input dim {dim}')
        plt.xlabel(f'input[{dim}]')
        plt.ylabel('model output')
        #plt.legend()
        plt.tight_layout()
        plt.savefig(f'sweeps/{model_name}_sweep_dim{dim}.png')
        plt.close()

