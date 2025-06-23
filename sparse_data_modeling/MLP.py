import numpy as np
import os

# Module‐level cache for loaded .npz files
_npz_cache: dict[str, np.lib.npyio.NpzFile] = {}

class MLP:
    def __init__(self, model_key: str):
        """
        model_key: string like 'corr_bb', 'corr_gg', etc.
        weights_file: path to the .npz generated above.
        """
        # Load (and cache) the .npz file
        weights_file = 'trained_models.npz'
        #add local path to the file -needs to work as a package
        weights_file = os.path.join(os.path.dirname(__file__), weights_file)

        data = _npz_cache.get(weights_file)
        if data is None:
            data = np.load(weights_file)
            _npz_cache[weights_file] = data

        # load all linear layers
        self.W1 = data[f'{model_key}_l1.weight']
        self.b1 = data[f'{model_key}_l1.bias']
        self.W2 = data[f'{model_key}_l2.weight']
        self.b2 = data[f'{model_key}_l2.bias']
        self.W3 = data[f'{model_key}_l3.weight']
        self.b3 = data[f'{model_key}_l3.bias']
        self.W4 = data[f'{model_key}_l4.weight']
        self.b4 = data[f'{model_key}_l4.bias']
        self.W_out = data[f'{model_key}_out.weight']
        self.b_out = data[f'{model_key}_out.bias']
        # c1, c2 are saved as zero-dimensional arrays
        self.c1 = float(data[f'{model_key}_c1'])
        self.c2 = float(data[f'{model_key}_c2'])

    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Element-wise ELU activation."""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: numpy array of shape (batch_size, input_dim)
        returns: numpy array of shape (batch_size, 1)
        """
        # Layer 1 numerator & denominator
        num1 = x.dot(self.W1.T) + self.b1
        den1 = self.elu(x.dot(self.W2.T) + self.b2) + 1.000001 + abs(self.c1)
        h1 = num1 / den1

        # Layer 2 numerator & denominator
        num2 = h1.dot(self.W3.T) + self.b3
        den2 = self.elu(h1.dot(self.W4.T) + self.b4) + 1.000001 + abs(self.c2)
        h2 = num2 / den2

        # Output
        out = np.tanh(h2.dot(self.W_out.T) + self.b_out)
        return out
