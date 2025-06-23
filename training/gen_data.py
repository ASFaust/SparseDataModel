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
n_datapoints = 200000 # number of total samples to generate in total
probs_per_abcd = 10 # number of p1,p2 pairs to sample per (a, b, c, d) sample
n_abcd_samples = n_datapoints // probs_per_abcd  # number of (a, b, c, d) samples to generate
#shuffle the abcd_samples to get a diverse set
np.random.shuffle(abcd_samples)
random_abcd_samples = abcd_samples[:n_abcd_samples]  # take only a subset for diversity

# Tracking seen keys
seen_keys = {
    'corr_bb': set(),
    'corr_gb': set(),
}

for i, sample in enumerate(random_abcd_samples):
    print(f"\rProcessing sample {i+1}/{len(random_abcd_samples)}: {sample}. \
corr_bb samples: {len(dataset['corr_bb'])}, \
corr_gg samples: {len(dataset['corr_gg'])}, \
corr_gb samples: {len(dataset['corr_gb'])}", end='', flush=True)

    lin = open_tanh_space(2.0, 20, 0.01) * 0.5 + 0.5
    p1p2 = list(product(lin, repeat=2))
    np.random.shuffle(p1p2)
    p1p2 = p1p2[:probs_per_abcd]
    corr = get_correlation_matrix(*sample)

    for p1, p2 in p1p2:
        th1 = norm.ppf(1.0 - p1)
        th2 = norm.ppf(1.0 - p2)
        samples = np.random.multivariate_normal(
            mean=np.zeros(4),
            cov=corr,
            size=sample_size
        )
        masks = samples[:, 2:] > np.array([th1, th2])
        values = samples[:, :2] * masks
        expanded_samples = np.hstack((values, masks))
        expanded_corr = np.corrcoef(expanded_samples, rowvar=False)

        key_bb = (p1, p2, corr[2, 3])
        key_gg = (p1, p2, corr[0, 1], corr[0, 3], corr[1, 2], corr[2, 3])
        key_gb = (p1, p2, corr[0, 3], corr[2, 3])

        if key_bb not in seen_keys['corr_bb']:
            seen_keys['corr_bb'].add(key_bb)
            dataset['corr_bb'].append(((p1, p2, expanded_corr[2, 3]), corr[2, 3]))

        dataset['corr_gg'].append(((p1, p2, expanded_corr[0, 1], expanded_corr[0, 3], expanded_corr[1, 2], expanded_corr[2, 3]), corr[0, 1]))

        if key_gb not in seen_keys['corr_gb']:
            seen_keys['corr_gb'].add(key_gb)
            dataset['corr_gb'].append(((p1, p2, expanded_corr[0, 3], expanded_corr[2, 3]), corr[0, 3]))

#save the dataset
import pickle
with open('training_data.pkl', 'wb') as f:
    pickle.dump(dataset, f)

