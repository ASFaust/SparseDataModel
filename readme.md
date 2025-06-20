# Modeling and Sampling of Sparsely distributed Data
In practical applications, data is often sparsely distributed across a domain. 
This can lead to challenges in modeling and sampling, as traditional methods may not be effective. 
In this repository, an approach to model and sample from sparsely distributed data is presented.

The data is assumed to be distributed according to a spike-and-slab prior: 
Each entry is either zero or drawn from a Gaussian distribution.
In such a case, naive computation of a covariance matrix and sampling from it does not 
retain the sparsity of the data, and underestimates the standard deviation of the non-zero entries.

## Approach
The approach involves the following steps:
* Inflate the dimensionality of the data by adding a new dimension for each entry, which represents wether the entry is zero or non-zero.
* Compute the pearson correlation matrix of the inflated data.
* 3 Different correction functions are applied to the pearson correlation matrix, depending on the index of the entry:
  * A correction function for the entries of correlations between the sparse gaussian data points
  * A correction function for the binary-binary interactions (Tetrachor correction)
  * A correction function for the binary-gaussian interactions 
* The corrected correlation matrix is then used to sample from a multivariate normal distribution.
* The entries corresponding to sparsity information are then thresholded to obtain binary entries
* The binary entries are then used to mask the sampled data, resulting in a sparse sample.

The correction functions are designed to retain the covariance structure: 
Reapplying this method to the sampled data will yield the same covariance structure as when applied to the original data.

This method also perfectly preserves the sparsity of the data, retaining the same statistical properties as the original data.
The correction functions are empirically derived from sampled data, and are not guaranteed to be optimal or theoretically justified.

there's also correction functions for the mean and std of the gaussians
which are applied separately
