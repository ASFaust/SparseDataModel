import numpy as np
from scipy.stats import norm

def get_correlation_matrix(a, b, c, d):
    return np.array([
        [1, a, 0, b],
        [a, 1, c, 0],
        [0, c, 1, d],
        [b, 0, d, 1]
    ])

def sample_observed_corr(corr, p1, p2, n):
    th1 = norm.ppf(1.0 - p1)
    th2 = norm.ppf(1.0 - p2)
    samples = np.random.multivariate_normal(mean=np.zeros(4), cov=corr, size=n)
    masks = samples[:, 2:] > np.array([th1, th2])
    values = samples[:, :2] * masks
    expanded = np.hstack((values, masks))
    return np.corrcoef(expanded, rowvar=False)

# config
n_points = 20
n_repeats = 100
n_samples = 20000 #20k is good enough!
p1, p2 = 0.9, 0.9

# load fixed abcds
abcds = np.load("valid_abcd_samples.npy")
selected = abcds[np.random.choice(len(abcds), size=n_points, replace=False)]

for i, abcd in enumerate(selected):
    a, b, c, d = abcd
    p1 = np.random.uniform(0.01, 0.99)
    p2 = np.random.uniform(0.01, 0.99)
    print(f"Processing point {i+1}/{n_points}: abcd={np.round(abcd,3)}, p1={p1:.2f}, p2={p2:.2f}")
    corr = get_correlation_matrix(a, b, c, d)
    vals = []
    for seed in range(n_repeats):
        C = sample_observed_corr(corr, p1, p2, n=n_samples)
        vals.append([
            C[2,3],  # binary-binary
            C[0,1],  # gaussian-gaussian
            C[0,3],  # gauss-binary
            C[1,2]   # gauss-binary
        ])
    vals = np.array(vals)
    stds = np.std(vals, axis=0)
    print(f"Point {i+1}: abcd={np.round(abcd,3)} -> std_bb={stds[0]:.4f}, std_gg={stds[1]:.4f}, std_gb0={stds[2]:.4f}, std_gb1={stds[3]:.4f}")
