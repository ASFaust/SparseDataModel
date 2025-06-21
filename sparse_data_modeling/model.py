import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
import os
from .mlp import MLP


def nearest_correlation_matrix(A):
    """Returns the nearest symmetric positive semi-definite matrix to A."""
    B = (A + A.T) / 2
    vals, vecs = np.linalg.eigh(B)
    vals = np.clip(vals, 1e-12, None)  # clamp small/negative eigenvalues
    ret = vecs @ np.diag(vals) @ vecs.T
    #normalize to get correlation matrix
    D_inv = np.diag(1.0 / np.sqrt(np.diag(ret)))
    corr = D_inv @ ret @ D_inv
    return corr

class SparseDataModel:
    """
    A sparse data model
    """
    def __init__(self, data : np.array):
        """
        Instantiate the model with some given sample data after which to model the distribution
        :param data: Expected to be of shape (n_samples, n_data_dims)
        """
        self.data = data
        #data has shape (n_samples, n_dims)
        self.n_samples, self.n_dims = data.shape


        torch.serialization.add_safe_globals({'MLPRegressor': MLP})
        model_path = os.path.join(os.path.dirname(__file__), 'trained_models.pt')
        trained_models = torch.load(model_path, weights_only=False)
        self.model_bb = trained_models['corr_bb'].cpu()
        self.model_gb = trained_models['corr_gb'].cpu()
        self.model_gg = trained_models['corr_gg'].cpu()
        self.model_bb.eval()
        self.model_gb.eval()
        self.model_gg.eval()

        self.mask = data != 0.0
        means = []
        stds = []
        for dim in range(self.n_dims):
            nonzero_data = data[self.mask[:, dim], dim]
            means.append(nonzero_data.mean())
            if nonzero_data.size <= 1:
                stds.append(1.0)
            else:
                stds.append(nonzero_data.std())

        self.means = np.array(means)
        self.stds = np.array(stds)
        normalized_data = np.where(self.mask, (data - self.means) / self.stds, 0.0)
        self.mask = self.mask.astype(np.float32)
        extended_data = np.concatenate((normalized_data, self.mask), axis=1)
        self.corr_naive = np.corrcoef(extended_data, rowvar=False)
        #replace nans with 0s in the correlation matrix
        self.corr_naive = np.nan_to_num(self.corr_naive, nan=0.0)
        #fill diagnonal with 1s
        np.fill_diagonal(self.corr_naive, 1.0)
        self.p = self.mask.mean(axis=0)
        self.thresholds = norm.ppf(1.0 - self.p)
        #now we get to the interesting part, we compute the corrected correlation matrix using our models
        self.corr = np.eye(self.n_dims * 2)
        #first we gotta do the binary-binary bit, which is the easiest.
        #it's the second half of the correlation matrix
        #input to the bb model:
        for i in range(self.n_dims):
            for j in range(self.n_dims):
                if i == j:
                    continue
                p_i, p_j = self.p[i], self.p[j]
                c_vivj = self.corr_naive[i, j]  # value–value
                c_vimj = self.corr_naive[i, j + self.n_dims]  # value_i – mask_j
                c_vjmi = self.corr_naive[j, i + self.n_dims]  # mask_i – value_j
                c_mimj = self.corr_naive[i + self.n_dims,
                                         j + self.n_dims]  # mask_i  – mask_j

                input_data = torch.tensor([[p_i, p_j, c_mimj]], dtype=torch.float32)
                c_bb = self.model_bb(input_data).item()
                self.corr[i + self.n_dims, j + self.n_dims] = c_bb #!! amazing, first one down!
                input_data = torch.tensor([[p_i, p_j, c_vivj, c_vimj, c_vjmi, c_mimj]], dtype=torch.float32)
                self.corr[i, j] = self.model_gg(input_data).item()
                input_data = torch.tensor([[p_i, p_j, c_vivj, c_vimj, c_mimj]], dtype=torch.float32)
                self.corr[i, j + self.n_dims] = self.model_gb(input_data).item()
                self.corr[j + self.n_dims, i] = self.corr[i, j + self.n_dims]  # symmetric part

        #correct the correlation matrix to be positive semi-definite
        self.corr = nearest_correlation_matrix(self.corr)


    def __call__(self,n_samples):
        """
        Generates n_samples from the model.
        Returns a numpy array of shape (n_samples, n_dims).
        """
        # Generate samples from the multivariate normal distribution
        samples = np.random.multivariate_normal(
            mean=np.zeros(self.n_dims * 2),
            cov=self.corr,
            size=n_samples
        )
        # Apply the thresholds to get binary values
        masks = samples[:, self.n_dims:] > self.thresholds
        values = samples[:, :self.n_dims] * self.stds + self.means
        values = np.where(masks, values, 0.0)  # replace masked values with 0

        return values

if __name__ == "__main__":
    # Example usage
    np.set_printoptions(precision=4, suppress=True)
    from random_model import RandomSparseDataModel
    n_dims = 4
    n_samples = 100000
    model = RandomSparseDataModel(n_dims)
    data = model(n_samples)
    sparse_data_model = SparseDataModel(data)

    # give me the original covariance matrix

    abs_diff = np.abs(model.corr - sparse_data_model.corr)
    print("Absolute difference between correlation matrices:")
    print(abs_diff.mean())
    # sample some data
    generated_data = sparse_data_model(n_samples)

    original_cov = np.cov(data, rowvar=False)
    generated_cov = np.cov(generated_data, rowvar=False)
    print("Original covariance matrix:")
    print(original_cov)
    print("Generated covariance matrix:")
    print(generated_cov)

    abs_diff = np.abs(original_cov - generated_cov)
    print("Absolute difference between covariance matrices:")
    print(abs_diff.mean())


