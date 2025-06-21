# SparseDataModel

A generative model for simulating high-dimensional sparse data under complex missingness patterns (including MNAR — Missing Not At Random). 
This model learns and reproduces both the joint distribution of values and their corresponding missingness structure using corrected 
correlation modeling via neural networks. 
No training is required, the model is initialized directly from a sample dataset and quickly generates synthetic data that matches the observed sparsity and correlation patterns.

## Features

- Learns statistical dependencies between observed values and missingness indicators
- Supports non-random (MNAR) missingness
- Corrects empirical correlations with learned regressors
- Generates realistic synthetic data matching observed sparsity and correlation patterns

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/ASFaust/SparseDataModel.git
cd SparseDataModel
pip install -e .
```

## Usage

```python
from sparse_data_model import SparseDataModel
sample_data = ... #sample_data needs to be a numpy array of shape (sample_dim,feature_dim)
# Initialize the model from sample data
model = SparseDataModel(sample_data)
# Generate synthetic data
synthetic_data = model.generate(num_samples=1000)
# synthetic_data has shape (1000, num_features)
```

## Applications

- Sparse CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- Simulating datasets with complex missingness patterns for research
- Benchmarking imputation algorithms under MNAR assumptions
- Data augmentation for training models robust to missing values
- Stress-testing statistical methods and pipelines under controlled sparsity
- Generating synthetic medical or sensor data where missingness is structurally dependent on observed variables
- Estimating uncertainty in downstream models trained on partially observed data
