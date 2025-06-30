import numpy as np
import os

from .MLP import MLP
from .norm_ppf import norm_ppf

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


        # Instantiate models with correct input dimensions
        self.model_bb = MLP('corr_bb')

        self.model_gb = MLP('corr_gb')

        self.model_gg = MLP('corr_gg')

        self.mask = data != 0.0
        means = []
        stds = []
        for dim in range(self.n_dims):
            nonzero_data = data[self.mask[:, dim], dim]
            if nonzero_data.size == 0:
                means.append(0.0)
            else:
                means.append(nonzero_data.mean())
            if nonzero_data.size <= 1:
                stds.append(1e-12)  # very small std to avoid division by zero
            else:
                std = nonzero_data.std()
                if np.isnan(std) or std == 0.0:
                    stds.append(1e-12)
                else:
                    stds.append(std)

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
        self.thresholds = norm_ppf(1.0 - self.p)
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
                c_mimj = self.corr_naive[i + self.n_dims, j + self.n_dims]  # mask_i  – mask_j

                input_data = np.array([[p_i, p_j, c_mimj]], dtype=np.float32)
                c_bb = self.model_bb.forward(input_data)[0, 0]
                self.corr[i + self.n_dims, j + self.n_dims] = c_bb

                input_data = np.array([[p_i, p_j, c_vivj, c_vimj, c_vjmi, c_mimj]], dtype=np.float32)
                self.corr[i, j] = self.model_gg.forward(input_data)[0, 0]

                input_data = np.array([[p_i, p_j, c_vivj, c_vimj, c_mimj]], dtype=np.float32)
                self.corr[i, j + self.n_dims] = self.model_gb.forward(input_data)[0, 0]
                self.corr[j + self.n_dims, i] = self.corr[i, j + self.n_dims]

        for dim in range(self.n_dims):
            if self.p[dim] == 0.0:
                self.means[dim] = 0.0
                self.stds[dim] = 1e-12  # very small std to avoid division by zero
                self.corr[dim, :] = 0.0
                self.corr[:, dim] = 0.0
                self.corr[dim + self.n_dims, :] = 0.0
                self.corr[:, dim + self.n_dims] = 0.0
                self.corr[dim, dim] = 1.0
                self.corr[dim + self.n_dims, dim + self.n_dims] = 1.0
            elif self.p[dim] == 1.0:
                # The value is handled fine by the model, but the mask is constant.
                # Set correlation of the constant mask with everything else to 0.
                mask_idx = dim + self.n_dims
                self.corr[mask_idx, :] = 0.0
                self.corr[:, mask_idx] = 0.0
                self.corr[mask_idx, mask_idx] = 1.0

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
