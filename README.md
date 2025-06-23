# SparseDataModel

A generative model for simulating sparse data that captures complex, non-random (MNAR) missingness patterns.

SparseDataModel learns the intricate statistical dependencies between data values and their missingness indicators directly from a sample dataset. It uses this understanding to generate new, synthetic data that preserves the original's sparsity structure and correlation patterns.

Crucially, the model requires no training. It is initialized directly from your data and can begin generating high-fidelity sparse samples immediately.

---
## Core Concepts

The model is built on two key ideas:

### Spike-and-Slab Distribution

The model conceptualizes your data with a spike-and-slab prior.

* The "spike" is a probability mass at exactly zero, representing values that are missing or truly zero.
* The "slab" is a Gaussian distribution that models the observed, non-zero values.

This joint approach allows the model to learn not just the distribution of the values, but the patterns of their presence or absence.

### Internal Modeling

SparseDataModel constructs a joint latent Gaussian model over both observed values and missingness indicators.

For a dataset with \(n\) features, the internal representation is a multivariate Gaussian over a \(2n\)-dimensional latent space:

- The first \(n\) dimensions correspond to the **data values** (the "slab").
- The next \(n\) dimensions correspond to the **missingness indicators** (the "spike"), modeled as thresholded latent Gaussians.

This results in a \(2n \times 2n\) **latent correlation matrix**:

- \(\Sigma_{X,X}\): correlations between data values.
- \(\Sigma_{M,M}\): correlations between missingness indicators.
- \(\Sigma_{X,M}\): cross-correlations between data values and missingness.
- \(\Sigma_{M,X} = \Sigma_{X,M}^T\): transpose of the cross-correlation matrix.

The mask variables are modeled using a **thresholded Gaussian** approach (e.g., probit), where a latent variable determines whether each entry is observed (non-zero) or missing (zero). This design enables the model to replicate MNAR behavior, where missingness depends on the values of other variables.

The joint correlation matrix can be inspected after initialization and is used internally for generating synthetic data that faithfully mirrors both the observed values and their sparsity structure.

### Pre-trained Neural Network Correctors

Sparsity and non-random missingness can distort the apparent correlations in a dataset. 
The model uses a set of pre-trained neural networks to correct these distortions upon initialization, 
using the naive pearson correlations computed from the input sample data.
These networks estimate the three key blocks of the latent correlation matrix:

* \(\Sigma_{X,X}\): the true correlation between the values themselves, as if they were fully observed (the "slab").
* \(\Sigma_{X,M}\): the correlation between the values and their missingness (the "spike" and "slab" interaction).
* \(\Sigma_{M,M}\): the true underlying correlation of the sparsity structure (the "spike").

This correction process ensures the generated data's correlation structure is a faithful representation of the underlying relationships in the input sample.

The network for the \(\Sigma_{X,X}\) block is a numerical approximation of the inverse of the tetrachoric transform,
mapping from observed binary correlations (between pairs of 0/1-masked variables) and their marginal probabilities
back to the underlying latent Gaussian correlations.
It inverts the joint thresholded bivariate normal model that governs the spike-and-slab process,
allowing the reconstruction of the original continuous correlation structure from sparsely observed binary data.

---

## Features

* **MNAR-Aware**: Explicitly models non-random missingness where the probability of a value being missing depends on other observed values.
* **No Training Required**: Initializes instantly from a sample numpy array, making it fast and easy to use.
* **Structure Preservation**: Generates synthetic data that matches the sparsity patterns and complex correlation structure of the input data.
* **Spike-and-Slab Foundation**: Robustly models data where missingness (== 0) and observed values (non-zero) are intertwined.

---

## Limitations


The model assumes that each feature is independent of its own missingness indicator. 
In other words, it does not model the correlation between a variable’s value and the probability of it being missing.

This is an intentional design choice. If a variable's own missingness depends on its value (e.g., younger individuals are more 
likely to omit their age), then the observed data no longer follows a Gaussian distribution. 
Instead, it's a truncated or censored distribution, for which estimating true mean and variance becomes impossible 
without additional assumptions or external data.

Modeling such self-missingness correlations would require learning not only new estimators for the correlation but also for 
the marginal distributions of the values themselves. These estimators tend to be unreliable, 
especially when the true correlation is weak or moderate. Empirical tests showed that trying to infer this information 
from the data directly leads to high uncertainty and poor generalization.

By setting self-missingness correlations to zero, the model can reliably estimate the joint structure of the 
rest of the data while avoiding strong assumptions about unobserved distributions. This simplification is what makes the 
model initialization robust and practical without needing training.

---

## Performance and Scalability

The primary computational cost occurs during model initialization.

The time complexity is approximately:

* \$O(n³)\$ in terms of the number of features, \$n\$, in the input data, due to the need to compute eigenvalues and eigenvectors for correlation matrices.
* \$O(m²)\$ in terms of the number of samples, \$m\$, in the input data, due to the need to compute pairwise correlations. 

Once the model is initialized, generating synthetic samples is very fast.

## Requirements

This package requires Python 3.8 or higher and `numpy` for numerical operations. Nothing else is needed to run the model.

If you wish to train your own neural networks for the correction step, you will also need `torch`.

---

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/ASFaust/SparseDataModel.git
cd SparseDataModel
pip install -e .
```

---

## Usage

```python
import numpy as np
from sparse_data_modeling import SparseDataModel

# sample_data should be a numpy array of shape (n_samples, n_features)
# Zeros are treated as missing values.
sample_data = np.random.randn(500, 50)
sample_data[np.random.rand(*sample_data.shape) > 0.8] = 0

model = SparseDataModel(sample_data)

print(model.corr) # numpy array of shape (1000, 1000), containing the corrected latent correlation matrix (elements in [-1, 1])
print(model.means) # numpy array of shape (1000,), containing the means of the nonzero values
print(model.stds) # numpy array of shape (1000,), containing the standard deviations of the nonzero values
print(model.p) # numpy array of shape (1000,), containing the probabilities of each feature being non-zero
print(model.thresholds) # numpy array of shape (1000,), containing the thresholds for the gaussians that model the missingness indicators
#the thresholds are equal to scipy.norm.ppf(1.0 - model.p) (scipy is not a dependency)

# Generate synthetic data
synthetic_data = model(n_samples=1000)

# synthetic_data has shape (1000, n_features)
# It shares the same MNAR patterns and correlation structure as sample_data.
```

---

## Applications

* **Enhanced Evolutionary Strategies**: Serve as a data-driven sampler for algorithms like Sparse CMA-ES.

---

## Future Work

Allow imputation of missing values in new data based on the learned distributions.
