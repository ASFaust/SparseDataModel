# SparseDataModel

A generative model for simulating sparse data under complex missingness patterns (including MNAR — Missing Not At Random). 
This model works with spike-and-slab priors to capture the joint distribution of observed values and their missingness indicators. 

Pretrained neural networks are used to correct empirical correlations,
ensuring that the generated data reflects the observed sparsity and correlation structure of the input sample data. 
For example, one network estimates the inverse tetrachoric correction for the sparsity structure (under biased thresholding),
while another network estimates something akin to point-biserial correction between the observed values and their missingness indicators.
A third network estimates the true correlation between the values if they were fully observed, correcting for the effects of correlated missingness.

No training is required, the model is initialized directly from a sample dataset and quickly generates synthetic data that matches the observed sparsity and correlation patterns.

## Features

- Learns statistical dependencies between observed values and missingness indicators
- Supports non-random (MNAR) missingness
- Corrects empirical correlations with neural networks 
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
from sparse_data_modeling import SparseDataModel
sample_data = ... #sample_data needs to be a numpy array of shape (sample_dim,feature_dim)
# Initialize the model from sample data
model = SparseDataModel(sample_data)
# Generate synthetic data
synthetic_data = model(n_samples=1000)
# synthetic_data has shape (1000, num_features)
# it has the same sparsity and correlation structure as sample_data, with MNAR patterns preserved
```

## Applications

- Sparse CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- Simulating datasets with complex missingness patterns for research
- Benchmarking imputation algorithms under MNAR assumptions
- Data augmentation for training models robust to missing values
- Stress-testing statistical methods and pipelines under controlled sparsity
- Generating synthetic medical or sensor data where missingness is structurally dependent on observed variables
- Estimating uncertainty in downstream models trained on partially observed data

## Future Work
- Allow imputation of missing values based on learned distributions
