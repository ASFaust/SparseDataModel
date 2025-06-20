import numpy as np
from scipy.stats import skew
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
        if nz.size > 1:
            m[i] = nz.mean()
            s[i] = nz.std(ddof=0)
            sk[i] = skew(nz)
        # else leave m, s, sk at zero

    # avoid division by zero when standardizing for corr
    sane = np.where(s == 0, 1.0, s)
    X = (samples - m) / sane
    R_naive = np.corrcoef(X, rowvar=False)
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

        mean_corr = np.zeros(n_dims)
        std_corr = np.zeros(n_dims)
        for v in range(n_dims):
            others = [k for k in range(n_dims) if k != v]
            row = R_naive[v, others]
            mean_corr[v] = np.nanmean(row)
            std_corr[v] = np.nanstd(row)

        # per-pair features for corr estimators
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue

                # then for each pair i≠j:
                x_base = [
                    R_naive[i, j],
                    p[i], p[j],
                    m[i], m[j],
                    s[i], s[j],
                    sk[i], sk[j],
                ]

                x = np.array(x_base + [
                    mean_corr[i], std_corr[i],
                    mean_corr[j], std_corr[j],
                ])
                datasets['corr_bb'].append((x, corr_true[i, j]))
                datasets['corr_gg'].append((x, corr_true[n_dims+i, n_dims+j]))
                datasets['corr_gb'].append((x, corr_true[n_dims+i, j]))

        # compute per-variable mean/std of off-diagonal correlations
        mean_corr = np.zeros(n_dims)
        std_corr  = np.zeros(n_dims)
        for i in range(n_dims):
            others = [j for j in range(n_dims) if j != i]
            row = R_naive[i, others]
            mean_corr[i] = np.nanmean(row)
            std_corr[i]  = np.nanstd(row)

        # per-variable features for mean/std/threshold estimators
        for i in range(n_dims):
            x = np.concatenate([
                [p[i]],
                [m[i], s[i], sk[i]],
                [mean_corr[i], std_corr[i]]
            ])
            datasets['mean'].append((x, mu_true[i]))
            datasets['std'].append((x, sigma_true[i]))
            datasets['threshold'].append((x, theta_true[i]))

    return datasets


# Example usage:
if __name__ == "__main__":
    data = generate_training_data(
        n_dims=10,
        n_samples_per_draw=100_000,
        n_draws=1000,
        seed=42
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