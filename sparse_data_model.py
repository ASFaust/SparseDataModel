import numpy as np

def random_correlation_matrix(n, seed=None):
    """
    Generates a random correlation matrix of size n x n.
    The matrix is symmetric, positive semi-definite, with 1s on the diagonal.
    """
    if seed is not None:
        np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Step 1: generate a random matrix
    A = rng.standard_normal((n, n))

    # Step 2: create a covariance matrix (symmetric and PSD)
    cov = np.dot(A, A.T)

    # Step 3: normalize to get correlation matrix
    D_inv = np.diag(1.0 / np.sqrt(np.diag(cov)))
    corr = D_inv @ cov @ D_inv

    return corr


class SparseDataGenerator:
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
        self.corr = random_correlation_matrix(n_dims * 2, seed=seed)
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
        masks = samples[:, :self.n_dims] > self.sparsity_thresholds
        values = samples[:, self.n_dims:] * self.nonzero_stds + self.nonzero_means
        return np.where(masks, values, 0.0)  # replace masked values with 0
