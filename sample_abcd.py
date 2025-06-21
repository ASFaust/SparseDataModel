import numpy as np
from itertools import product

"""
# Define the matrix
M = sp.Matrix([
    [1, a, 0, b],
    [a, 1, c, 0],
    [0, c, 1, d],
    [b, 0, d, 1]
])
"""

def determinant_constraint(a, b, c, d):
    #the above matrix has this determinant:
    #and if this is >= 0, then the matrix is positive semidefinite
    return a**2 * d**2 - a**2 - 2 * a * b * c * d + b**2 * c**2 - b**2 - c**2 - d**2 + 1

def eigval_constraint(a, b, c, d):
    M = np.array([
        [1, a, 0, b],
        [a, 1, c, 0],
        [0, c, 1, d],
        [b, 0, d, 1]
    ])
    # Check if the eigenvalues of M are all non-negative
    eigvals = np.linalg.eigvalsh(M)
    return np.all(eigvals >= 0)

def tanh_space(scale, grid_resolution):
    #tanh space is a squashing of the linspace to oversample boundary regions
    lin = np.linspace(-scale, scale, grid_resolution)
    max_val = np.tanh(scale)
    min_val = np.tanh(-scale)
    tanh_vals = np.tanh(lin)
    tanh_vals = (tanh_vals - min_val) / (max_val - min_val) * 2 - 1  # scale to [-1, 1]
    return tanh_vals

def grid_sample_valid_abcd(grid_resolution=25):
    lin = tanh_space(2.0, grid_resolution)  # tanh space for better coverage near boundaries
    samples = []
    n_invalid = 0
    for a, b, c, d in product(lin, repeat=4):
        if determinant_constraint(a, b, c, d) >= 0:
            if eigval_constraint(a, b, c, d):
                # If both constraints are satisfied, add the sample
                samples.append((a, b, c, d))
            else:
                n_invalid += 1
        else:
            n_invalid += 1
    print(f"\nGenerated {len(samples)} valid points out of {grid_resolution**4} total grid points, {n_invalid} invalid.")
    print(f"Coverage: {len(samples) / (grid_resolution**4) * 100:.2f}%")
    return np.array(samples)

if __name__ == "__main__":
    n_samples = 10000
    valid_samples = grid_sample_valid_abcd()
    print(f"Generated {len(valid_samples)} valid (a, b, c, d) samples.")

    # Optionally save to a file or process further
    np.save('valid_abcd_samples.npy', valid_samples)
    print("Samples saved to 'valid_abcd_samples.npy'.")