"""
Uses SparseDataModel as the sampling distribution of a simple elitist EDA on shifted test functions
with a sparsity penalty: minimize f(x - x_star) + penalty * count_nonzero(x).
The optimum x_star is sparse, so the EDA has to find both its support and its non-zero values.
"""
import numpy as np
import pytest

from sparse_data_modeling import SparseDataModel

N_DIMS = 10
PENALTY = 0.5
X_STAR = np.zeros(N_DIMS)
X_STAR[[1, 4, 7]] = [1.5, -2.0, 2.5]
OPTIMAL_FITNESS = PENALTY * np.count_nonzero(X_STAR)
SEEDS = range(5)


def sphere(z):
    return np.sum(z ** 2, axis=-1)


def ellipsoid(z):
    n = z.shape[-1]
    return np.sum(10.0 ** (3.0 * np.arange(n) / (n - 1)) * z ** 2, axis=-1)


def rosenbrock(z):
    y = z + 1.0  # shifted so that the optimum is at z = 0
    return np.sum(100.0 * (y[..., 1:] - y[..., :-1] ** 2) ** 2 + (1.0 - y[..., :-1]) ** 2, axis=-1)


def rastrigin(z):
    return 10.0 * z.shape[-1] + np.sum(z ** 2 - 10.0 * np.cos(2.0 * np.pi * z), axis=-1)


def penalized(f):
    return lambda x: f(x - X_STAR) + PENALTY * np.count_nonzero(x, axis=-1)


def sparse_eda(fitness, n_dims, n_generations=80, pop_size=200, elite_size=40, std_inflation=1.5, init_p=0.5, seed=0):
    """
    Elitist EDA: keep the elite_size best samples, fit a SparseDataModel to them and sample the rest.
    The slab stds are inflated before sampling to counter the premature convergence of truncation selection.
    The initial population is spike and slab with non-zero probability init_p.
    Returns the best sample, its fitness and the fitness of the best initial sample.
    """
    np.random.seed(seed)  # SparseDataModel samples from the global numpy RNG
    rng = np.random.default_rng(seed)
    population = rng.normal(0.0, 2.0, (pop_size, n_dims)) * (rng.random((pop_size, n_dims)) < init_p)
    scores = fitness(population)
    initial_best = scores.min()

    for _ in range(n_generations):
        elite_idx = np.argsort(scores)[:elite_size]
        elite, elite_scores = population[elite_idx], scores[elite_idx]

        model = SparseDataModel(elite, prob_eps=0.02, diagonal_lambda=0.05, std_floor=1e-8)
        model.stds = model.stds * std_inflation
        offspring = model(pop_size - elite_size)

        population = np.vstack([elite, offspring])
        scores = np.concatenate([elite_scores, fitness(offspring)])

    best = np.argmin(scores)
    return population[best], scores[best], initial_best


@pytest.mark.parametrize("f", [sphere, ellipsoid, rosenbrock], ids=lambda f: f.__name__)
def test_finds_sparse_optimum(f):
    fitness = penalized(f)
    n_solved = 0
    for seed in SEEDS:
        x, fx, _ = sparse_eda(fitness, N_DIMS, seed=seed)
        support_found = np.array_equal(x != 0, X_STAR != 0)
        if support_found and fx - OPTIMAL_FITNESS < 1e-6:
            np.testing.assert_allclose(x, X_STAR, atol=1e-3)
            n_solved += 1
    # the support search is discrete and can get stuck, so allow a failing run
    assert n_solved >= len(SEEDS) - 1


def test_multimodal_function_reaches_sparse_local_optimum():
    """
    Rastrigin traps the EDA in local optima that drop part of the support, so only check that the
    result improves, has no spurious non-zeros and that the kept coordinates are optimal.
    """
    fitness = penalized(rastrigin)
    for seed in SEEDS:
        x, fx, initial_best = sparse_eda(fitness, N_DIMS, seed=seed)
        assert fx < initial_best
        kept = x != 0
        assert not np.any(kept & (X_STAR == 0))
        np.testing.assert_allclose(x[kept], X_STAR[kept], atol=1e-3)


def test_dense_optimum():
    """No sparsity in the optimum: all masks become always-on and the model must handle p == 1."""
    # every |x_star_i| > sqrt(PENALTY), so a non-zero value is optimal in every dim
    x_star = np.where(np.arange(N_DIMS) % 2 == 0, 1.0, -1.0) * np.linspace(1.0, 2.0, N_DIMS)
    fitness = lambda x: sphere(x - x_star) + PENALTY * np.count_nonzero(x, axis=-1)
    x, fx, _ = sparse_eda(fitness, N_DIMS, n_generations=150, init_p=1.0, seed=0)
    assert np.count_nonzero(x) == N_DIMS
    np.testing.assert_allclose(x, x_star, atol=1e-3)
