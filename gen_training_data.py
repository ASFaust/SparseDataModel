import numpy as np
from collections import defaultdict

def compute_naive_stats(samples):
    """
    samples: (n_samples, n_dims) observed (zero-inflated) data
    Returns:
      p: shape (n_dims,) nonzero probability
      m: shape (n_dims,) mean of nonzeros
      s: shape (n_dims,) std of nonzeros
      sk: shape (n_dims,) skewness of nonzeros
      R_naive: (n_dims, n_dims) Pearson corr of observed data
    """
    n_samples, n_dims = samples.shape
    p = (samples != 0).mean(axis=0)
    m = np.zeros(n_dims)
    s = np.zeros(n_dims)
    sk = np.zeros(n_dims)

    for i in range(n_dims):
        xi = samples[:, i]
        nz = xi[xi != 0]
        #if nz.size > 1:
        m[i] = nz.mean()
        s[i] = nz.std()
        # for skewness, we use the proportion of values greater than the mean
        sk[i] = (np.mean(nz > m[i]) - 0.5) if nz.size > 0 else 0.0

    # we need to extend samples to include 2*ndim: include an indicator dim for nonzero
    samples_extended = np.hstack((samples, (samples != 0).astype(float)))
    R_naive = np.corrcoef(samples_extended, rowvar=False)
    return p, m, s, sk, R_naive

def generate_training_data(
    n_dims,
    n_samples_per_draw,
    n_draws,
    seed=None
):
    """
    Runs SparseDataGenerator n_draws times, each time:
      1. draws n_samples_per_draw samples
      2. computes naive stats & true params
      3. collects feature / target pairs for all 6 estimators
    Returns:
      dict of datasets, each is a list of (X_row, y_true) pairs
      keys: 'corr_bb', 'corr_gb', 'corr_gg',
            'mean', 'std', 'threshold'
    """
    from sparse_data_model import SparseDataGenerator  # adjust as needed

    datasets = defaultdict(list)

    for draw in range(n_draws):
        print(f"\rGenerating draw {draw + 1}/{n_draws}...", end='', flush=True)
        gen = SparseDataGenerator(n_dims, seed=(seed or 0) + draw)
        corr_true = gen.corr
        mu_true = gen.nonzero_means
        sigma_true = gen.nonzero_stds
        theta_true = gen.sparsity_thresholds

        samples = gen(n_samples_per_draw)
        p, m, s, sk, R_naive = compute_naive_stats(samples)

        # per-pair features for corr estimators
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue

                # then for each pair i≠j:
                x = [
                    R_naive[i, j],
                    R_naive[n_dims+i, n_dims+j],
                    R_naive[n_dims+i, j],
                    R_naive[i, n_dims+j],
                    p[i], p[j],
                    m[i], m[j],
                    s[i], s[j],
                    sk[i], sk[j],
                ]
                x = np.array(x)
                datasets['corr_bb'].append((x, corr_true[i, j]))
                datasets['corr_gg'].append((x, corr_true[n_dims+i, n_dims+j]))
                datasets['corr_gb'].append((x, corr_true[n_dims+i, j]))

        # per-variable features for mean/std/threshold estimators
        for i in range(n_dims):
            x = np.array([
                p[i], m[i], s[i], sk[i]
            ])
            datasets['mean'].append((x, mu_true[i]))
            datasets['std'].append((x, sigma_true[i]))
            datasets['threshold'].append((x, theta_true[i]))

    return datasets


# Example usage:
if __name__ == "__main__":
    data = generate_training_data(
        n_dims=2,
        n_samples_per_draw=50_000,
        n_draws=100,
        seed=1294
    )
    # The returned data is a dictionary with keys:
    # 'corr_bb', 'corr_gb', 'corr_gg', 'mean', 'std', 'threshold'
    # data['corr_bb'] is a list of (feature_vector, true_corr) pairs, etc.
    # You can now split into X/y and train your 6 regressors.
    for key in data:
        print(f"{key}: {len(data[key])} samples")
        # Optionally, you can save or process the data further
    # print an example output
    for key in data:
        print(f"{key} example: {data[key][0]}")
        print(f"dimensionality of input features: {len(data[key][0][0])}")

    #save to file
    import pickle
    with open('training_data.pkl', 'wb') as f:
        pickle.dump(data, f)