from gen_training_data import generate_training_data

data = generate_training_data(
    n_generators=10,
    n_samples=1000,
    n_dims=5,
    seed_base=42
)

print(data)