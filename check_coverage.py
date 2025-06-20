import numpy as np
import pickle

def check_coverage(datasets, bins=10):
    """
    Assess coverage of 6D input space by binning and checking fill ratio.
    Assumes all datasets have the same feature dimensionality (6).
    """
    from itertools import product

    print("\n--- Coverage Check ---")
    key = next(iter(datasets))
    data = datasets[key]
    X = np.array([x for x, _ in data])
    assert X.shape[1] == 6, "Expected 6D input vectors."

    # Bin edges based on known input ranges: 4 in [-1,1], 2 in [0,1]
    edges = [
        np.linspace(-1, 1, bins + 1),
        np.linspace(-1, 1, bins + 1),
        np.linspace(-1, 1, bins + 1),
        np.linspace(-1, 1, bins + 1),
        np.linspace(0, 1, bins + 1),
        np.linspace(0, 1, bins + 1)
    ]

    # Digitize each dimension
    digitized = [
        np.digitize(X[:, dim], edges[dim]) - 1  # 0-indexed bin numbers
        for dim in range(6)
    ]
    digitized = np.stack(digitized, axis=1)
    digitized = np.clip(digitized, 0, bins - 1)

    # Count unique bin occupancies
    bins_seen = set(tuple(row) for row in digitized)
    total_bins = bins ** 6
    occupied_bins = len(bins_seen)
    sparsity_ratio = occupied_bins / total_bins

    print(f"Total possible bins: {total_bins}")
    print(f"Occupied bins:       {occupied_bins}")
    print(f"Coverage:            {sparsity_ratio:.6f} ({sparsity_ratio*100:.4f}%)")

    # Show 10 random unoccupied bins
    # Show 10 random unoccupied bin value centers
    all_bins = list(product(range(bins), repeat=6))
    unoccupied_bins = list(set(all_bins) - bins_seen)

    print("\n10 random unoccupied input vectors (bin centers):")
    centers = []
    for _ in range(10):
        idx = np.random.randint(len(unoccupied_bins))
        bin_coords = unoccupied_bins[idx]
        center = []
        for d in range(6):
            lo = edges[d][bin_coords[d]]
            hi = edges[d][bin_coords[d] + 1]
            center.append((lo + hi) / 2.0)
        centers.append(center)

    for i, c in enumerate(centers, 1):
        print(f"{i:2d}: {np.round(c, 4)}")

    return sparsity_ratio


if __name__ == "__main__":
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)

    check_coverage(data, bins=4)