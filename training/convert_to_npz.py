#!/usr/bin/env python3
import torch
import numpy as np
from train_models import MLP

torch.serialization.add_safe_globals({"MLP": MLP})
# Load the dict of trained models
models = torch.load('trained_models.pt', map_location='cpu', weights_only=False)

# Prepare a flat dict of numpy arrays
np_weights = {}
for model_name, model in models.items():
    state = model.state_dict()
    for param_name, tensor in state.items():
        # key in npz will be e.g. "corr_bb_l1.weight"
        key = f'{model_name}_{param_name}'
        np_weights[key] = tensor.cpu().numpy()

# Save everything into one .npz
np.savez('trained_models.npz', **np_weights)
print("Saved all model parameters to trained_models.npz")
