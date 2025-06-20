import numpy as np
from collections import defaultdict
import tqdm

def compute_naive_stats(samples):
    """
    Vectorized version of naive stats computation.
    """
    n_samples, n_dims = samples.shape
    indicator = (samples != 0).astype(float)
    p = indicator.mean(axis=0)

    # mask: (n_samples, n_dims), True where nonzero
    mask = indicator.astype(bool)

    # set zeros to NaN temporarily for mean/std computation
    samples_masked = samples.copy()
    samples_masked[~mask] = np.nan

    # compute mean and std over nonzero values using nanmean/nanstd
    m = np.nanmean(samples_masked, axis=0)
    s = np.nanstd(samples_masked, axis=0)

    # normalize nonzero entries
    samples_normalized = np.where(mask, (samples - m) / s, 0.0)

    # append indicator mask for correlation computation
    samples_extended = np.hstack((samples_normalized, indicator))
    R_naive = np.corrcoef(samples_extended, rowvar=False)

    return p, R_naive

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
    counting_bins = np.zeros((200,),dtype = np.int32)  # for keeping track of wether the data is balanced

    pbar = tqdm.tqdm(total=n_draws, desc="Generating draws", unit="draw")

    for draw in range(n_draws):
        random_seed = np.random.randint(0, 2**31 - 1)
        pbar.update(1)
        #print(f"\rGenerating draw {draw + 1}/{n_draws}...", end='', flush=True)
        gen = SparseDataGenerator(n_dims, seed=(seed or random_seed) + draw)
        corr_true = gen.corr
        mu_true = gen.nonzero_means
        sigma_true = gen.nonzero_stds
        theta_true = gen.sparsity_thresholds
        #now plop all off diagonal correlation values into the bins
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                corr_values = [
                    corr_true[i, j],
                    corr_true[n_dims + i, n_dims + j],
                    corr_true[i, n_dims + j]
                ]
                for corr_value in corr_values:
                    bin_index = int((corr_value + 1.0) * 100.0)
                    counting_bins[bin_index] += 1


        samples = gen(n_samples_per_draw)
        if draw == 0:
            #samples has shape (n_samples_per_draw, n_dims)
            data = samples[:,0]
            nonzero_mask = data != 0
            nonzero_data = data[nonzero_mask]
            estimated_mean = np.mean(nonzero_data)
            estimated_std = np.std(nonzero_data)
            true_mean = mu_true[0]
            true_std = sigma_true[0]
            import matplotlib.pyplot as plt
            plt.hist(nonzero_data, bins=50, density=True)
            #also plot the 2 normal distributions
            x = np.linspace(min(nonzero_data), max(nonzero_data), 100)
            plt.plot(x, (1 / (estimated_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - estimated_mean) / estimated_std) ** 2), label='Estimated Normal')
            plt.plot(x, (1 / (true_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - true_mean) / true_std) ** 2), label='True Normal')
            plt.legend()
            plt.title(f"Draw {draw + 1} - Nonzero Data Distribution")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.savefig(f'draw_{draw + 1}_histogram.png')

        values, R_naive = compute_naive_stats(samples)

        # per-pair features for corr estimators
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                x = [
                    R_naive[i,j],
                    R_naive[n_dims+i, n_dims+j],
                    R_naive[n_dims+i, j],
                    R_naive[i, n_dims+j],
                    values[i],  values[j],  # p[i], p[j]
                ]
                x = np.array(x)
                datasets['corr_bb'].append((x, corr_true[i, j]))
                datasets['corr_gg'].append((x, corr_true[n_dims+i, n_dims+j]))
                datasets['corr_gb'].append((x, corr_true[n_dims+i, j])) #fine with i!=j, because if i==j, then corr_true[n_dims+i, j] is 0.0

    #plot the counting bins
    print("\nCounting bins for correlation values:")
    import matplotlib.pyplot as plt
    #clear the figure
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(-1, 1, 0.01), counting_bins, width=0.01)
    plt.title('Counting Bins for Correlation Values')

    plt.xlabel('Correlation Value')
    plt.ylabel('Count')
    plt.savefig('counting_bins.png')

    return datasets

# Example usage:
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    data = generate_training_data(
        n_dims=10,
        n_samples_per_draw=1_000_000,
        n_draws=100,
        seed=None
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