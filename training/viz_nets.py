from os import pipe2

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from train_models import MLP
from scipy.stats import norm


np.set_printoptions(precision=4, suppress=True)
torch.serialization.add_safe_globals({"MLP": MLP})
# Load trained models
trained_models = torch.load("trained_models.pt", weights_only=False)
for model in trained_models.values():
    model.cpu()
    model.eval()

n_points = 500
sample_size = 20000 #number of samples for label creation
n_sweeps = 10 #number of sweeps to perform


os.makedirs("sweeps", exist_ok=True)

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

def get_correlation_matrix(a, b, c, d):
    return np.array([
        [1, a, 0, b],
        [a, 1, c, 0],
        [0, c, 1, d],
        [b, 0, d, 1]
    ])

def eigval_constraint(a, b, c, d):
    M = get_correlation_matrix(a, b, c, d)
    # Check if the eigenvalues of M are all non-negative
    eigvals = np.linalg.eigvalsh(M)
    return np.all(eigvals >= 0)


def find_longest_true_run(mask):
    """
    Given a 1D boolean array, returns (start_idx, end_idx) of the longest
    contiguous True segment.
    """
    max_len = curr_len = 0
    max_start = curr_start = 0

    for i, ok in enumerate(mask):
        if ok:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
            if curr_len > max_len:
                max_len = curr_len
                max_start = curr_start
        else:
            curr_len = 0

    return max_start, max_start + max_len - 1  # both inclusive

def get_continuous_spd_segment(abcd_start, abcd_end):
    """
    Samples n_points along the line from abcd_start to abcd_end,
    finds the longest contiguous block for which eigval_constraint is True,
    and returns new endpoints for that block.
    """
    # create all sample points
    alphas = np.linspace(0.0, 1.0, n_points)
    all_abcd = np.array([abcd_start + alpha*(abcd_end - abcd_start)
                         for alpha in alphas])

    # test SPD‐validity
    mask = np.array([eigval_constraint(*pt) for pt in all_abcd])

    # find longest run
    i0, i1 = find_longest_true_run(mask)
    if i1 < i0:
        raise ValueError("No SPD point found in the sweep.")

    # corresponding alpha‐values
    alpha0, alpha1 = alphas[i0], alphas[i1]

    # new segment endpoints
    new_start = abcd_start + alpha0*(abcd_end - abcd_start)
    new_end   = abcd_start + alpha1*(abcd_end - abcd_start)

    return new_start, new_end

def get_valid_abcd():
    # generate your original endpoints
    abcd_start = np.random.uniform(0, 1, 4)
    abcd_start /= np.max(np.abs(abcd_start))
    abcd_end   = np.random.uniform(-1, 0, 4)
    abcd_end   /= np.max(np.abs(abcd_end))

    # clamp to the longest SPD‐valid subsegment
    a0, b0, c0, d0 = get_continuous_spd_segment(abcd_start, abcd_end)[0]
    a1, b1, c1, d1 = get_continuous_spd_segment(abcd_start, abcd_end)[1]

    # now do your final sweep within [start,end]
    a_range = np.linspace(a0, a1, n_points)
    b_range = np.linspace(b0, b1, n_points)
    c_range = np.linspace(c0, c1, n_points)
    d_range = np.linspace(d0, d1, n_points)
    return np.array(list(zip(a_range, b_range, c_range, d_range)))

def get_model_input(p1,p2,abcd):
    a, b, c, d = abcd
    original_corr = np.array([
        [1, a, 0, b],
        [a, 1, c, 0],
        [0, c, 1, d],
        [b, 0, d, 1]
    ])
    corr = nearest_correlation_matrix(original_corr)
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
    #expanded_corr is now the input to the models.
    return expanded_corr

#colors = plt.cm.viridis(np.linspace(0, 1, n_samples))

ylabel_dict = {
    'corr_bb': 'Binary-Binary Correlation (d)',
    'corr_gg': 'Value-Value Correlation (a)',
    'corr_gb': 'Value-Mask Correlation (b)'
}

for j in range(n_sweeps):
    abcd = get_valid_abcd()
    p1 = np.random.uniform(0.01, 0.99)
    p2 = np.random.uniform(0.01, 0.99)
    p = np.array([p1, p2])
    sample_corrs = []
    for i in range(n_points):
        print(f"\rProcessing point {i+1}/{n_points} for sweep {j+1}: p1={p1:.2f}, p2={p2:.2f}, abcd={abcd[i]}", end='', flush=True)
        sample_corr = get_model_input(p1, p2, abcd[i])
        sample_corrs.append(sample_corr)
    print()
    results = [] #results contains a list of uncorrected correlation matrices
    print("running models...")

    #what i want to do here is:
    #for each point, reconstruct the correlation matrix from the model input
    pred_as = []
    pred_bs = []
    pred_cs = []
    pred_ds = []
    raw_as = []
    raw_bs = []
    raw_cs = []
    raw_ds = []
    corrected_mats = []
    for i in range(n_points):
        corr_naive = sample_corrs[i]
        output_corr = np.eye(4)  # initialize the output correlation matrix
        #this loop is for reconstructing the original correlation matrix by running the models for each element of n_points
        #so in the innermost loop, 1 call to corr_bb, 1 call to corr_gg, and 2 calls to corr_gb
        for d1 in range(2):
            for d2 in range(2):
                if d1 == d2:
                    continue
                input_bb = torch.tensor([[p[d1], p[d2], corr_naive[d1 + 2, d2 + 2]]], dtype=torch.float32)
                input_gg = torch.tensor([[p[d1], p[d2], corr_naive[d1, d2], corr_naive[d1 + 2, d2], corr_naive[d1, d2 + 2], corr_naive[d1 + 2, d2 + 2]]], dtype=torch.float32)

                output_corr[d1, d2] = trained_models['corr_gg'](input_gg).item()
                output_corr[d1 + 2, d2 + 2] = trained_models['corr_bb'](input_bb).item()

                input_gb1 = torch.tensor([[p[d1], p[d2], corr_naive[d1, d2 + 2], corr_naive[d1 + 2, d2 + 2]]], dtype=torch.float32)

                output_corr[d1, d2 + 2] = trained_models['corr_gb'](input_gb1).item()

                input_gb2 = torch.tensor([[p[d2], p[d1], corr_naive[d1 + 2, d2], corr_naive[d1 + 2, d2 + 2]]],
                                         dtype=torch.float32)
                output_corr[d1 + 2, d2] = trained_models['corr_gb'](input_gb2).item()

        corrected_corr = nearest_correlation_matrix(output_corr)
        #print(corrected_corr)
        true_corr = get_correlation_matrix(*abcd[i])
        #print(f"True correlation matrix:\n{true_corr}")
        #compute frobenius norm of the difference
        diff = np.linalg.norm(corrected_corr - true_corr, ord='fro')
        if diff > 0.9:
            print(f"Warning: Large difference in sweep {j+1}, point {i+1}: {diff:.4f}")
            print(f"Parameters: {abcd[i]}, p1={p1:.2f}, p2={p2:.2f}")
            print(f"True correlation matrix:\n{true_corr}")
            print(f"Predicted correlation matrix:\n{output_corr}")
            print(f"Corrected correlation matrix:\n{corrected_corr}")
        pred_as.append(corrected_corr[0, 1])
        pred_bs.append(corrected_corr[0, 3])
        pred_cs.append(corrected_corr[1, 2])
        pred_ds.append(corrected_corr[2, 3])
        corrected_mats.append(corrected_corr.copy())
        raw_as.append(output_corr[0, 1])
        raw_bs.append(output_corr[0, 3])
        raw_cs.append(output_corr[1, 2])
        raw_ds.append(output_corr[2, 3])

    params = [
        ('a', abcd[:, 0], np.array(pred_as), np.array(raw_as)),
        ('b', abcd[:, 1], np.array(pred_bs), np.array(raw_bs)),
        ('c', abcd[:, 2], np.array(pred_cs), np.array(raw_cs)),
        ('d', abcd[:, 3], np.array(pred_ds), np.array(raw_ds)),
    ]
    print("plotting results...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.ravel()

    for ax, (name, true_vals, pred_vals, raw_vals) in zip(axes[:4], params):
        x_indices = np.arange(len(true_vals))
        ax.scatter(x_indices, pred_vals - true_vals, alpha=0.5, edgecolors='none', label='Corrected Predicted Values', color='blue')
        ax.scatter(x_indices, raw_vals - true_vals, alpha=0.5, marker='x', label='Raw Predicted Values', color='orange')
        #ax.plot(x_indices, true_vals, color='red', label='True Values', linewidth=2, linestyle='--')
        #ax.set_ylim(-1,1)
        ax.set_title(f'Parameter {name}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Prediction Error')

    #then also plot the frobenius norm of the difference between true and corrected correlation matrices
    ax = axes[4]
    for i in range(n_points):
        true_corr = get_correlation_matrix(*abcd[i])
        corrected_corr = corrected_mats[i]
        diff = np.linalg.norm(corrected_corr - true_corr, ord='fro')
        ax.scatter(i, diff, color='green', alpha=0.5)

    ax = axes[5]
    for i in range(n_points):
        true_corr = get_correlation_matrix(*abcd[i])
        corrected_corr = corrected_mats[i]
        val_true = np.linalg.norm(true_corr, ord='fro')
        val_pred = np.linalg.norm(corrected_corr, ord='fro')
        ax.scatter(i, val_true, color='red', alpha=0.5, label='True Correlation Norm')
        ax.scatter(i, val_pred, color='blue', alpha=0.5, label='Predicted Correlation Norm')

    fig.suptitle('True vs. Predicted Correlation Parameters', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(f'sweeps/sweep_{j+1}_true_vs_predicted.png', dpi= 450)
    plt.close(fig)









