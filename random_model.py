import numpy as np

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

class RandomSparseDataModel:
    """
    This generator generates data distributed as sparse spike and slab data with random means and probabilities
    """
    def __init__(self,n_dims,seed=None):
        """

        :param n_dims:
        :param seed:
        we first generate a random 2n x 2n correlation matrix
        """
        self.n_dims = n_dims
        self.beta = 0.5 #np.random.rand() + 1e-6
        self.corr = np.random.beta(self.beta, self.beta, size=(n_dims * 2, n_dims * 2)) * 2.0 - 1.0
        for i in range(n_dims):
            self.corr[i, i + n_dims] = 0.0
            self.corr[i + n_dims, i] = 0.0
            self.corr[i, i] = 1.0  # ensure diagonal is 1
        self.corr = nearest_correlation_matrix(self.corr)
        #zero out self correlation between the values and its own mask
        for j in range(10):
            for i in range(n_dims):
                self.corr[i, i + n_dims] = 0.0
                self.corr[i + n_dims, i] = 0.0
                self.corr[i,i] = 1.0  # ensure diagonal is 1
            #ensure the correlation matrix is positive semi-definite
            self.corr = nearest_correlation_matrix(self.corr)

        self.nonzero_means = (np.random.rand(n_dims) * 2.0 - 1.0) * 2.0
        self.nonzero_stds = (np.random.rand(n_dims) * 2.0) * 2.0
        self.sparsity_thresholds = (np.random.rand(n_dims) * 2.0 - 1.0) * 2.0

    def __call__(self, n_samples):
        """
        Generate n_samples of data
        :param n_samples:
        :return: a numpy array of shape (n_samples, n_dims)
        """
        #generate a random sample from the multivariate normal distribution with the given correlation matrix
        samples = np.random.multivariate_normal(
            mean=np.zeros(self.n_dims * 2),
            cov=self.corr,
            size=n_samples
        )
        masks = samples[:, self.n_dims:] > self.sparsity_thresholds
        values = samples[:, :self.n_dims]
        values = values * self.nonzero_stds + self.nonzero_means
        values = np.where(masks, values, 0.0)
        return values  # replace masked values with 0
