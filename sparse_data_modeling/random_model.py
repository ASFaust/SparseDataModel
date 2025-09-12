import numpy as np
from scipy.stats import norm

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

def exp_decay_corr_with_mask(n_dims, decay=0.9, n_iter=5, seed=None):
    """
    Generate a random correlation matrix for 2*n_dims variables,
    using an exponential decay eigenvalue spectrum, with enforced
    zeros between each variable and its own mask dimension.

    Parameters
    ----------
    n_dims : int
        Number of value dimensions (total size = 2*n_dims).
    decay : float in (0,1)
        Decay factor for eigenvalue spectrum.
        - decay ~ 1.0 -> flat spectrum (weak correlations).
        - decay small -> steep spectrum (strong correlations).
    n_iter : int
        Iterations of (mask enforce + PSD projection).
    seed : int or None
        Random seed.

    Returns
    -------
    corr : ndarray, shape (2*n_dims, 2*n_dims)
        Correlation matrix with mask structure.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_dims

    # 1. exponential spectrum
    eigs = decay ** np.arange(n)
    eigs = eigs * (n / eigs.sum())  # normalize to sum = n

    # 2. random orthogonal basis
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    M = Q @ np.diag(eigs) @ Q.T

    # 3. normalize to correlation
    D_inv = np.diag(1.0 / np.sqrt(np.diag(M)))
    corr = D_inv @ M @ D_inv

    # 4. enforce mask structure and re-project
    for _ in range(n_iter):
        for i in range(n_dims):
            corr[i, i+n_dims] = 0.0
            corr[i+n_dims, i] = 0.0
            corr[i, i] = 1.0
            corr[i+n_dims, i+n_dims] = 1.0

        # nearest PSD projection
        vals, vecs = np.linalg.eigh((corr + corr.T) / 2)
        vals = np.clip(vals, 1e-8, None)
        corr = vecs @ np.diag(vals) @ vecs.T
        D_inv = np.diag(1.0 / np.sqrt(np.diag(corr)))
        corr = D_inv @ corr @ D_inv

    return corr



class RandomSparseDataModel:
    """
    This generator generates data distributed as sparse spike and slab data with random means and probabilities
    """
    def __init__(self,n_dims, decay=0.8, max_p = 0.1, seed=None):
        """

        :param n_dims:
        :param seed:
        we first generate a random 2n x 2n correlation matrix
        """
        self.n_dims = n_dims
        self.corr = exp_decay_corr_with_mask(n_dims, decay=decay, seed=seed)
        self.means = (np.random.rand(n_dims) * 2.0 - 1.0) * 2.0
        self.stds = (np.random.rand(n_dims) * 2.0) * 2.0
        #self.sparsity_thresholds = (np.random.rand(n_dims) * 2.0 - 1.0) * 2.0
        #random_p should be in (max_p, 1-max_p)
        self.p = np.random.rand(n_dims)  * (1.0 - 2.0 * max_p) + max_p
        #then the thresholds are the inverse cdf of the standard normal at 1 - random_p
        self.sparsity_thresholds = norm.ppf(1.0 - self.p)

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
        values = values * self.stds + self.means
        values = np.where(masks, values, 0.0)
        return values  # replace masked values with 0
