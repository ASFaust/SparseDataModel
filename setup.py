from setuptools import setup, find_packages

setup(
    name="sparse_data_modeling",               # package name on PyPI (if you publish)
    version="1.0.0",
    description="Sparse data covariance model",
    author="Andreas Faust",
    author_email="andreas.s.faust@gmail.com",
    packages=find_packages(),        # finds 'sparse_cov'
    install_requires=[
        "numpy",
        "scipy",
        "torch"
    ],
    python_requires=">=3.8",         # adjust as needed
)
