import numpy as np
import pickle
from itertools import product
from scipy.stats import norm
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Load samples
abcd_samples = np.load('valid_abcd_samples.npy')

# Constants
sample_size = 100000
probs_per_abcd = 10
random_abcd_samples = abcd_samples

def open_tanh_space(scale, grid_resolution, openness=0.01):
    lin = np.linspace(-scale, scale, grid_resolution)
    max_val = np.tanh(scale) + openness
    min_val = np.tanh(-scale) - openness
    tanh_vals = np.tanh(lin)
    tanh_vals = (tanh_vals - min_val) / (max_val - min_val) * 2 - 1
    return tanh_vals

def get_correlation_matrix(a, b, c, d):
    return np.array([
        [1, a, 0, b],
        [a, 1, c, 0],
        [0, c, 1, d],
        [b, 0, d, 1]
    ])

def process_abcd(sample):
    lin = open_tanh_space(2.0, 20, 0.01) * 0.5 + 0.5
    p1p2 = list(product(lin, repeat=2))
    np.random.shuffle(p1p2)
    p1p2 = p1p2[:probs_per_abcd]
    corr = get_correlation_matrix(*sample)

    results = {
        'corr_bb': [],
        'corr_gg': [],
        'corr_gb': []
    }

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

        results['corr_bb'].append(((p1, p2, expanded_corr[2, 3]), corr[2, 3], key_bb))
        results['corr_gg'].append(((p1, p2, expanded_corr[0, 1], expanded_corr[0, 3], expanded_corr[1, 2], expanded_corr[2, 3]), corr[0, 1], key_gg))
        results['corr_gb'].append(((p1, p2, expanded_corr[0, 3], expanded_corr[2, 3]), corr[0, 3], key_gb))

    return results

def merge_results(result_list):
    merged = {
        'corr_bb': [],
        'corr_gg': [],
        'corr_gb': [],
    }
    seen_keys = {
        'corr_bb': set(),
        'corr_gg': set(),
        'corr_gb': set(),
    }
    for result in result_list:
        for k in merged:
            for datapoint in result[k]:
                x, label, key = datapoint
                if key not in seen_keys[k]:
                    seen_keys[k].add(key)
                    merged[k].append((x, label))
    return merged

if __name__ == '__main__':
    n_cpus = cpu_count()
    n_cpus = min(n_cpus, 16)  # Limit to 16 CPUs for performance reasons
    print(f"Using {n_cpus} CPUs for processing.")
    with Pool(n_cpus) as pool:
        all_results = list(tqdm(pool.imap_unordered(process_abcd, random_abcd_samples),
                                total=len(random_abcd_samples),
                                desc="Processing"))

    dataset = merge_results(all_results)

    with open('training_data.pkl', 'wb') as f:
        pickle.dump(dataset, f)
