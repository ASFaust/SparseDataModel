#this script should discover symmetries in the data
import numpy as np
from scipy.stats import norm

def get_matrix(latent):
    a, b, c, d, p1, p2 = latent
    corr = np.array([
        [1, a, 0, b],
        [a, 1, c, 0],
        [0, c, 1, d],
        [b, 0, d, 1]
    ])
    return corr

def get_data(latent,sample_size):
    a, b, c, d, p1, p2 = latent
    corr = get_matrix(latent)
    if not is_spd(corr):
        return None
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
    #what we want to discover is not symmetries in abcd, but in expanded_corr:
    #which different expanded_corr values map to the same (a, b, c, d) values?
    o_a = expanded_corr[0, 1]
    o_b = expanded_corr[0, 3]
    o_c = expanded_corr[1, 2]
    o_d = expanded_corr[2, 3]
    ret = np.array([o_a, o_b, o_c, o_d, p1, p2], dtype=np.float32)
    #assert that none are nan
    assert not np.any(np.isnan(ret)), "Generated data contains NaN values"
    return ret

def transform(data, mask):
    c_mask = mask.astype(np.float32)  # Convert mask to float for multiplication
    ret = np.array([
        data[0] * (1.0 - 2.0 * c_mask[0]),  # a
        data[1] * (1.0 - 2.0 * c_mask[1]),  # b
        data[2] * (1.0 - 2.0 * c_mask[2]),  # c
        data[3] * (1.0 - 2.0 * c_mask[3]),  # d
        1.0 - data[4] if mask[4] else data[4],  # p1
        1.0 - data[5] if mask[5] else data[5] # p2
    ], dtype=np.float32)
    return ret

def create_all_masks(masks=[]):
    #create all masks of length 6 with 0-2 True values
    if len(masks) > 0:
        return masks
    for i in range(64):  # 2^6 = 64 possible masks
        mask = [(i >> j) & 1 for j in range(6)]
        masks.append(np.array(mask, dtype=bool))
    return masks

def is_spd(matrix):
    """Check if a matrix is symmetric positive definite."""
    if not np.allclose(matrix, matrix.T):
        return False  # Not symmetric
    try:
        np.linalg.cholesky(matrix)
        return True  # Positive definite
    except np.linalg.LinAlgError:
        return False  # Not positive definite

def get_symmetries(latent, eps, sample_size):
    masks = create_all_masks()
    all_observed = []  # Initialize list to hold observed data
    for i,mask in enumerate(masks):
        all_observed.append(get_data(transform(latent, mask), sample_size))
    target_observed = get_data(transform(latent, masks[0]), sample_size)  # Get the target observed data
    assert target_observed is not None, "The first mask should always yield valid data"
    symmetries = []
    for i, observed in enumerate(all_observed): 
        if observed is None: # Skip if the mask created non-spd matrix
            continue
        for j, mask2 in enumerate(masks):
            transformed_observed = transform(observed, mask2)
            dist = np.abs(transformed_observed - target_observed).sum()
            if dist < eps:
                symmetries.append((i,j))
    return symmetries

def nearest_correlation_matrix(A):
    """Returns the nearest symmetric positive semi-definite matrix to A."""
    B = (A + A.T) / 2
    vals, vecs = np.linalg.eigh(B)
    vals = np.clip(vals, 1e-12, None)  # clamp small/negative eigenvalues
    ret = vecs @ np.diag(vals) @ vecs.T
    #normalize to get correlation matrix
    D_inv = np.diag(1.0 / np.sqrt(np.diag(ret)))
    corr = D_inv @ ret @ D_inv
    return corr

def get_random_latent():
    while True:
        a,b,c,d = np.random.uniform(-1, 1, 4)  # Generate random a, b, c, d
        p1, p2 = np.random.uniform(0.1, 0.9, 2)
        M = get_matrix((a, b, c, d, p1, p2))
        ret = np.array((M[0, 1], M[0, 3], M[1, 2], M[2, 3], p1, p2), dtype=np.float32)

        for i in range(100):  # Ensure the correlation matrix is positive semi-definite
            M = get_matrix(ret)
            M = nearest_correlation_matrix(M)  # Ensure M is a valid correlation matrix
            ret = np.array((M[0, 1], M[0, 3], M[1, 2], M[2, 3], p1, p2), dtype=np.float32)
        if is_spd(get_matrix(ret)):
            break
    return ret

def print_transformation(mask):
    print_str = f"a *= {-1.0 if mask[0] else 1.0}, "
    print_str += f"b *= {-1.0 if mask[1] else 1.0}, "
    print_str += f"c *= {-1.0 if mask[2] else 1.0}, "
    print_str += f"d *= {-1.0 if mask[3] else 1.0}, "
    print_str += f"p1 = {"1.0 - p1" if mask[4] else "p1"}, "
    print_str += f"p2 = {"1.0 - p2" if mask[5] else "p2"}"
    print(print_str)

def find_symmetries(n_trials=1000, eps=1e-1, sample_size=10000):
    symmetry_proof = {}  # Initialize symmetry proof dictionary
    for _ in range(n_trials):
        print(f"\rTrial {_ + 1}/{n_trials}", end="", flush=True)
        latent = get_random_latent()
        symmetries = get_symmetries(latent, eps, sample_size)
        for sym in symmetries:
            if sym not in symmetry_proof:
                symmetry_proof[sym] = 0
            symmetry_proof[sym] += 1
    # Filter out symmetries that were not found in any trial
    #print("Symmetry proof counts:", symmetry_proof)
    #print sorted by number of occurrences
    sorted_symmetries = sorted(symmetry_proof.items(), key=lambda x: x[1], reverse=True)
    print("Symmetries found (mask index, count):")
    all_masks = create_all_masks()
    for sym, count in sorted_symmetries:
        if count > n_trials * 0.9: # Only print symmetries found in more than half of the trials
            print(f"{sym}: {count} occurrences, ")
            #we need to print the actual transformation
            mask1 = all_masks[sym[0]]
            mask2 = all_masks[sym[1]]
            print("observed gets transformed this way:")
            print_transformation(mask2)
            print("and the resulting latent gets transformed this way:")
            print_transformation(mask1)
            print("-" * 50)

np.set_printoptions(precision=3, suppress=True)

find_symmetries(n_trials=100, eps=0.2, sample_size=10000)

