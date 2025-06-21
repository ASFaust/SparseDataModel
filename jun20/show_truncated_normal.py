import numpy as np

from sparse_data_model import random_correlation_matrix

corr_matrix = random_correlation_matrix(2, seed=None)
print(corr_matrix)
#just 1 random threshold value
threshold = np.random.rand(1)[0] * 2.0 - 1.0
n_samples = 10000
samples = np.random.multivariate_normal(
    mean=np.zeros(2),
    cov=corr_matrix,
    size=n_samples
)
#samples has shape (n_samples, 2)
mask = samples[:, 1] > threshold
values = samples[:, 0]
masked_values = np.where(mask, values, 0.0)  # replace masked values with 0
nonzero_masked_values = masked_values[masked_values != 0.0]
#plot histogram of nonzero_masked_values
import matplotlib.pyplot as plt
plt.hist(nonzero_masked_values, bins=100, density=True)
plt.title('Histogram of Nonzero Masked Values')
plt.xlabel('Value')
plt.ylabel('Density')
plt.savefig('histogram_nonzero_masked_values.png')
