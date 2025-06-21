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

### Pre-trained Neural Network Correctors

Sparsity and non-random missingness can distort the apparent correlations in a dataset. 
The model uses a set of pre-trained neural networks to correct these distortions upon initialization:

* They estimate the true underlying correlation of the sparsity structure (the "spike").
* They estimate the correlation between the values and their missingness (the "spike" and "slab" interaction).
* They estimate the true correlation between the values themselves, as if they were fully observed (the "slab").

This correction process ensures the generated data's correlation structure is a faithful representation of the underlying relationships in the input sample.

---

## Features

* **MNAR-Aware**: Explicitly models non-random missingness where the probability of a value being missing depends on other observed values.
* **No Training Required**: Initializes instantly from a sample numpy array, making it fast and easy to use.
* **Structure Preservation**: Generates synthetic data that matches the sparsity patterns and complex correlation structure of the input data.
* **Spike-and-Slab Foundation**: Robustly models data where missingness (== 0) and observed values (non-zero) are intertwined.

---

## Performance and Scalability

The primary computational cost occurs during model initialization, specifically when finding the nearest valid correlation matrix.

* The time complexity is \$\mathcal{O}(n^3)\$, where \$n\$ is the number of features (i.e., the dimensionality) of the data, and
* \$O(m²)\$ in terms of the number of samples, \$m\$, in the input data.
* Once the model is initialized, generating synthetic samples is very fast.

The `nearest_correlation_matrix` function, which dominates initialization time, relies on eigenvalue decomposition (`np.linalg.eigh`), an \$\mathcal{O}(n^3)\$ operation.

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
# e.g., sample_data = np.random.randn(500, 50)
# sample_data[np.random.rand(*sample_data.shape) > 0.8] = 0

sample_data = ...  # Initialize the model from sample data
model = SparseDataModel(sample_data)

# Generate synthetic data
synthetic_data = model(n_samples=1000)

# synthetic_data has shape (1000, n_features)
# It shares the same MNAR patterns and correlation structure as sample_data.
```

---

## Applications

* **Algorithm Benchmarking**: Generate realistic test cases for imputation algorithms under MNAR conditions.
* **Robustness Testing**: Stress-test statistical methods and ML pipelines against controlled, complex missingness.
* **Data Augmentation**: Increase the size of a sparse dataset for training models that are robust to missing values.
* **Simulation**: Create synthetic datasets (e.g., medical or sensor data) where missingness is structurally dependent on observed variables.
* **Uncertainty Estimation**: Analyze the impact of data sparsity on the uncertainty of downstream models.
* **Enhanced Evolutionary Strategies**: Serve as a data-driven sampler for algorithms like Sparse CMA-ES.

---

## Future Work

Allow imputation of missing values in new data based on the learned distributions.
