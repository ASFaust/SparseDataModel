from sparse_data_modeling import SparseDataModel
from sparse_data_modeling import RandomSparseDataModel
import numpy as np
import torch

# Example usage
np.set_printoptions(precision=4, suppress=True)
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
