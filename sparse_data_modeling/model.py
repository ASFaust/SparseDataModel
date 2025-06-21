import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
import os

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

        model_path = os.path.join(os.path.dirname(__file__), 'trained_model_weights.pt')
        state_dicts = torch.load(model_path, map_location='cpu')

        # Instantiate models with correct input dimensions
        self.model_bb = MLP(input_dim=3)
        self.model_bb.load_state_dict(state_dicts['corr_bb'])

        self.model_gb = MLP(input_dim=5)
        self.model_gb.load_state_dict(state_dicts['corr_gb'])

        self.model_gg = MLP(input_dim=6)
        self.model_gg.load_state_dict(state_dicts['corr_gg'])

        # Set to evaluation mode
        self.model_bb.eval()
        self.model_gb.eval()
        self.model_gg.eval()

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
                stds.append( 1e-12)  # very small std to avoid division by zero
            else:
                std = nonzero_data.std()
                if np.isnan(std) or std == 0.0:
                    stds.append( 1e-12)
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
                c_mimj = self.corr_naive[i + self.n_dims, j + self.n_dims]  # mask_i  – mask_j

                input_data = torch.tensor([[p_i, p_j, c_mimj]], dtype=torch.float32)
                c_bb = self.model_bb(input_data).item()
                self.corr[i + self.n_dims, j + self.n_dims] = c_bb #!! amazing, first one down!
                input_data = torch.tensor([[p_i, p_j, c_vivj, c_vimj, c_vjmi, c_mimj]], dtype=torch.float32)
                self.corr[i, j] = self.model_gg(input_data).item()
                input_data = torch.tensor([[p_i, p_j, c_vivj, c_vimj, c_mimj]], dtype=torch.float32)
                self.corr[i, j + self.n_dims] = self.model_gb(input_data).item()
                self.corr[j + self.n_dims, i] = self.corr[i, j + self.n_dims]  # symmetric part

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

if __name__ == "__main__":
    # Example usage
    np.set_printoptions(precision=4, suppress=True)
    from random_model import RandomSparseDataModel
    n_dims = 5
    n_samples = 1000
    model = RandomSparseDataModel(n_dims)
    model.sparsity_thresholds[-1] = -float('inf')  # make the last dimension always on
    model.nonzero_means[-2] = 4.0
    model.nonzero_stds[-2] = 1e-13  # very small std to avoid division by zero
    model.sparsity_thresholds[-2] = 0.0 # 50% chance of being on, but if it is on, it is always 4.0
    data = model(n_samples)
    #data has shape (n_samples, n_dims)
    #we add a constantly 0 dimension to the data
    data = np.hstack((data, np.zeros((n_samples, 1))))  # add a constant dimension

    sparse_data_model = SparseDataModel(data)

    # give me the original covariance matrix
    sparse_corr = sparse_data_model.corr
    # it has shape (n_dims * 2, n_dims * 2)
    # we need to extend it to ((n_dims + 1) * 2, (n_dims + 1) * 2)
    # by adding a row and column of zeros
    #sparse_corr = np.pad(sparse_corr, ((0, 1), (0, 1)), mode='constant', constant_values=0.0)

    #abs_diff = np.abs(model.corr - sparse_corr)
    #print("Absolute difference between correlation matrices:")
    #print(abs_diff.mean())
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
    print(abs_diff.min(), abs_diff.mean(), abs_diff.max())


    #measure frobenius norm of the difference
    frobenius_norm = np.linalg.norm(original_cov - generated_cov, 'fro')
    print("Frobenius norm of the difference between covariance matrices:")
    print(frobenius_norm)


    #measure squared 2-wasserstein distance
    #trace(sigma1 + sigma2 - 2 * sqrt(sqrt(sigma1) * sigma2 * sqrt(sigma1)))
    from scipy.linalg import sqrtm
    def wasserstein_distance_squared(sigma1, sigma2):
        """Compute squared 2-Wasserstein distance between two covariance matrices."""
        sqrt_sigma1 = sqrtm(sigma1)
        sqrt_sigma2 = sqrtm(sigma2)
        term = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
        return np.trace(sigma1 + sigma2 - 2 * sqrtm(term))

    wasserstein_dist_sq = wasserstein_distance_squared(original_cov, generated_cov)
    print("Squared 2-Wasserstein distance between the two covariance matrices:")
    print(wasserstein_dist_sq)
