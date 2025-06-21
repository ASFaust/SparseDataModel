import numpy as np
from itertools import product
from scipy.stats import norm

#load from valid_abcd_samples.npy

def open_tanh_space(scale, grid_resolution, openness=0.01):
    #tanh space is a squashing of the linspace to oversample boundary regions
    lin = np.linspace(-scale, scale, grid_resolution)
    max_val = np.tanh(scale) + openness
    min_val = np.tanh(-scale) - openness
    tanh_vals = np.tanh(lin)
    tanh_vals = (tanh_vals - min_val) / (max_val - min_val) * 2 - 1  # scale to [-1, 1]
    return tanh_vals

"""
# Define the matrix
M = sp.Matrix([
    [1, a, 0, b],
    [a, 1, c, 0],
    [0, c, 1, d],
    [b, 0, d, 1]
])
"""

def get_correlation_matrix(a, b, c, d):
    # then we can instantiate the process which samples the data
    # fill the diagonal with 1s
    return np.array([
        [1, a, 0, b],
        [a, 1, c, 0],
        [0, c, 1, d],
        [b, 0, d, 1]
    ])


abcd_samples = np.load('valid_abcd_samples.npy')

print(f"Loaded {len(abcd_samples)} valid (a, b, c, d) samples from 'valid_abcd_samples.npy'.")
np.set_printoptions(precision=2, suppress=True)

dataset  = {
    'corr_bb': [],
    'corr_gg': [],
    'corr_gb': [],
}


#sample_size = 20000  # 20k samples is a good number for training, determined with some experimentation (estimate_label_noise.py)
sample_size = 100000 # 100k samples for high quality training data
n_datapoints = 20000 # number of total samples to generate in total
probs_per_abcd = 1 # number of p1,p2 pairs to sample per (a, b, c, d) sample
n_abcd_samples = n_datapoints // probs_per_abcd  # number of (a, b, c, d) samples to generate
#shuffle the abcd_samples to get a diverse set
np.random.shuffle(abcd_samples)
random_abcd_samples = abcd_samples[:n_abcd_samples]  # take only a subset for diversity

#we now additionally need to cover the probabilites p1,p2 which are in [0,1]
for i,sample in enumerate(random_abcd_samples):
    print(f"\rProcessing sample {i+1}/{len(random_abcd_samples)}: {sample}", end='', flush=True)
    lin = open_tanh_space(2.0, 20, 0.01) * 0.5 + 0.5  # scale to [0, 1]
    #choose 2 random from lin
    p1p2 = product(lin, repeat=2)
    #choose 4 random pairs from p1p2
    p1p2 = list(p1p2)
    np.random.shuffle(p1p2)
    p1p2 = p1p2[:probs_per_abcd]  # take only a few pairs for diversity
    corr = get_correlation_matrix(*sample)
    for p1,p2 in p1p2:
        #thresholds are norm.ppf(1.0 - p1) and norm.ppf(1.0 - p2
        th1 = norm.ppf(1.0 - p1)
        th2 = norm.ppf(1.0 - p2)
        #print(f"Sample: {sample}, p1: {p1}, p2: {p2}, thresholds: {th1}, {th2}")
        #now we can generate the data
        samples = np.random.multivariate_normal(
            mean=np.zeros(4),
            cov=corr,
            size=sample_size  # 20k samples is a good number for training, determined with some experimentation (estimate_label_noise.py)
        )
        #apply the thresholds
        masks = samples[:, 2:] > np.array([th1, th2])
        values = samples[:, :2] * masks
        #now we have the values and masks
        expanded_samples = np.hstack((values, masks))
        #then determine the correlation matrix of the expanded samples
        expanded_corr = np.corrcoef(expanded_samples, rowvar=False)

        #so now we already have all the data we need to train our models
        #we just gotta save it properly
        #binary-binary interactions just need p1, p2, and the binary-binary correlation value
        #gaussian-gaussian interactions need all 6 values: binary-binary, gaussian-gaussian, binary-gaussian (x2), and p1, p2
        #gaussian-binary interactions need all 6 values: binary-binary, gaussian-gaussian, binary-gaussian (x2), and p1, p2

        dataset['corr_bb'].append(((p1, p2, expanded_corr[2,3]), corr[2,3]))
        dataset['corr_gg'].append(((p1, p2, expanded_corr[0,1], expanded_corr[0,3], expanded_corr[1,2], expanded_corr[2,3]), corr[0,1]))
        dataset['corr_gb'].append(((p1, p2, expanded_corr[0,1], expanded_corr[0,3], expanded_corr[2,3]), corr[0,3]))

#save the dataset
import pickle
with open('training_data.pkl', 'wb') as f:
    pickle.dump(dataset, f)

