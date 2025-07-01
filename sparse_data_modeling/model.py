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
    # normalize to get correlation matrix
    D_inv = np.diag(1.0 / np.sqrt(np.diag(ret)))
    corr = D_inv @ ret @ D_inv
    return corr


class SparseDataModel:
    """
    A sparse data model that learns a distribution from data with missing values.
    """

    def __init__(self, data: np.array):
        """
        Instantiate and fit the model based on the given sample data.

        :param data: A numpy array of shape (n_samples, n_data_dims) with missing values represented as 0.0.
        """
        # --- Basic Setup ---
        self.data = data
        self.n_samples, self.n_dims = data.shape
        self.mask = (self.data != 0.0).astype(np.float32)

        # --- Model Initialization and Computation Steps ---
        self._initialize_models()
        self.means, self.stds = self._calculate_statistics()
        self.p, self.thresholds = self._compute_thresholds()
        naive_corr = self._compute_naive_correlation()

        #and in this step we can also use the symmetries we discovered
        self.corr = self._compute_corrected_correlation(naive_corr)

        self._handle_correlation_edge_cases()
        self._ensure_positive_semidefinite()

    def _initialize_models(self):
        """Initializes the MLP models for correlation correction."""
        self.model_bb = MLP('corr_bb')
        self.model_gb = MLP('corr_gb')
        self.model_gg = MLP('corr_gg')

    def _calculate_statistics(self):
        """
        Calculates the mean and standard deviation for each dimension using only the non-zero values.
        """
        means = []
        stds = []
        for dim in range(self.n_dims):
            nonzero_data = self.data[self.mask[:, dim].astype(bool), dim]

            if nonzero_data.size == 0:
                means.append(0.0)
                stds.append(1e-12)  # Default for empty data
            elif nonzero_data.size == 1:
                means.append(nonzero_data.mean())
                stds.append(1e-12)  # Small std to avoid division by zero
            else:
                means.append(nonzero_data.mean())
                std = nonzero_data.std()
                stds.append(std if std > 1e-12 else 1e-12)

        return np.array(means), np.array(stds)

    def _compute_naive_correlation(self):
        """
        Computes the initial, uncorrected correlation matrix from the normalized data and masks.
        """
        normalized_data = np.where(self.mask.astype(bool), (self.data - self.means) / self.stds, 0.0)
        extended_data = np.concatenate((normalized_data, self.mask), axis=1)

        corr = np.corrcoef(extended_data, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def _compute_thresholds(self):
        """
        Computes the probability of non-zero values (p) and the corresponding Gaussian thresholds.
        """
        p = self.mask.mean(axis=0)
        thresholds = norm_ppf(1.0 - p)
        return p, thresholds

    def _compute_corrected_correlation(self, naive_corr):
        """
        Uses the MLP models to compute the corrected correlation matrix from the naive correlations.
                if key_bb not in seen_keys['corr_bb']:
            seen_keys['corr_bb'].add(key_bb)
            dataset['corr_bb'].append(((p1, p2, expanded_corr[2, 3]), corr[2, 3]))

        dataset['corr_gg'].append(((p1, p2, expanded_corr[0, 1], expanded_corr[0, 3], expanded_corr[1, 2], expanded_corr[2, 3]), corr[0, 1]))

        if key_gb not in seen_keys['corr_gb']:
            seen_keys['corr_gb'].add(key_gb)
            dataset['corr_gb'].append(((p1, p2, expanded_corr[0, 3], expanded_corr[2, 3]), corr[0, 3]))

        """
        corr = np.eye(self.n_dims * 2)
        for i in range(self.n_dims):
            for j in range(self.n_dims):  # Iterate over upper triangle
                if i == j:
                    continue
                p_i, p_j = self.p[i], self.p[j]

                # Extract relevant naive correlations
                c_vivj = naive_corr[i, j]
                c_vimj = naive_corr[i, j + self.n_dims]
                c_vjmi = naive_corr[j, i + self.n_dims]
                c_mimj = naive_corr[i + self.n_dims, j + self.n_dims]

                # Binary-Binary correction
                bb_in = np.array([[p_i, p_j, c_mimj]], dtype=np.float32)
                c_bb = self.model_bb.forward(bb_in)[0, 0]
                corr[i + self.n_dims, j + self.n_dims] = c_bb

                # Gaussian-Gaussian correction
                gg_in = np.array([[p_i, p_j, c_vivj, c_vimj, c_vjmi, c_mimj]], dtype=np.float32)
                c_gg = self.model_gg.forward(gg_in)[0, 0]
                corr[i, j] = c_gg

                # Gaussian-Binary correction
                gb_in = np.array([[p_i, p_j, c_vimj, c_mimj]], dtype=np.float32)
                c_gb = self.model_gb.forward(gb_in)[0, 0]
                corr[i, j + self.n_dims] = c_gb
                corr[j + self.n_dims, i] = c_gb

        return corr

    def _handle_correlation_edge_cases(self):
        """
        Post-processes the correlation matrix to handle columns that are all-zero or all-nonzero.
        """
        for dim in range(self.n_dims):
            # Handle columns that are completely empty (all zero)
            if self.p[dim] == 0.0:
                self.corr[dim, :] = 0.0
                self.corr[:, dim] = 0.0
                self.corr[dim + self.n_dims, :] = 0.0
                self.corr[:, dim + self.n_dims] = 0.0
                self.corr[dim, dim] = 1.0
                self.corr[dim + self.n_dims, dim + self.n_dims] = 1.0
            # Handle columns that are completely full (no zeros)
            elif self.p[dim] == 1.0:
                # The mask is constant, so its correlation with everything else should be 0.
                mask_idx = dim + self.n_dims
                self.corr[mask_idx, :] = 0.0
                self.corr[:, mask_idx] = 0.0
                self.corr[mask_idx, mask_idx] = 1.0

    def _ensure_positive_semidefinite(self):
        """
        Ensures the final correlation matrix is symmetric and positive semi-definite.
        """
        self.corr = nearest_correlation_matrix(self.corr)

    def __call__(self, n_samples):
        """
        Generates n_samples from the learned model.

        :param n_samples: The number of samples to generate.
        :return: A numpy array of shape (n_samples, n_dims).
        """
        # Generate samples from the multivariate normal distribution
        mean_vec = np.zeros(self.n_dims * 2)
        samples = np.random.multivariate_normal(mean=mean_vec, cov=self.corr, size=n_samples)

        # Apply thresholds to get the mask
        masks = samples[:, self.n_dims:] > self.thresholds

        # Denormalize values and apply the mask
        values = samples[:, :self.n_dims] * self.stds + self.means
        values = np.where(masks, values, 0.0)

        return values